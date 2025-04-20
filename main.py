import os
import numpy as np
import tqdm
import pandas as pd
import argparse
from typing import Union, Dict
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import torch
import torch.utils.data as data
import torch.optim as optim
import torchmetrics
import torchvision
import pytorch_lightning as pl
from click.core import F
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from sympy.integrals.risch import NonElementaryIntegral
from torch import classes
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, auc
from data import get_datasets, TransformTensorDataset
from model import ShallowCNN
from ood import score_fn
os.environ["TORCH_SHOW_CPP_STACKTRACES"] = "1"
torch.set_float32_matmul_precision('medium')
import torch.nn.functional as F
import random

def parse_args():
    parser = argparse.ArgumentParser(description='Configure training/inference/sampling for EBMs')
    parser.add_argument('--data_dir', type=str, default="/proj/aimi-adl/GLYPHS/",
                        help='path to directory with glyph image data')
    parser.add_argument('--ckpt_dir', type=str, default="/proj/ciptmp/ur03ocab/adl_ex3_files/",
                        help='path to directory where model checkpoints are stored')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--num_epochs', type=int, default=120,
                        help='number of epochs to train (default: 120)')
    parser.add_argument('--cbuffer_size', type=int, default=128,
                        help='num. images per class in the sampling reservoir (default: 128)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='learning rate (default: 1e-4)')
    parser.add_argument('--lr_gamma', type=float, default=0.97,
                        help='exponentional learning rate decay factor (default: 0.97)')
    parser.add_argument('--lr_stepsize', type=int, default=2,
                        help='learning rate decay step size (default: 2)')
    parser.add_argument('--alpha', type=int, default=0.1,
                        help='strength of L2 regularization (default: 0.1)')
    parser.add_argument('--num_classes', type=int, default=42,
                        help='number of output nodes/classes (default: 1 (EBM), 42 (JEM))')
    parser.add_argument('--ccond_sample', type=bool, default=True,
                        help='flag that specifies class-conditional or unconditional sampling (default: false')
    parser.add_argument('--num_workers', type=int, default="0",
                        help='number of loading workers, needs to be 0 for Windows')
    return parser.parse_args()

class MCMCSampler:
    def __init__(self, model, img_shape, sample_size, num_classes, cbuffer_size=256):
        super().__init__()
        self.model = model
        self.img_shape = img_shape
        self.sample_size = sample_size
        self.num_classes = num_classes
        self.cbuffer_size = cbuffer_size
        
        self.buffer = {i: torch.randn((cbuffer_size, *img_shape)) for i in range(num_classes)}

    def synthesize_samples(self, clabel=None, steps=60, step_size=10, return_img_per_step=False):  
        is_training = self.model.training
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        had_gradients_enabled = torch.is_grad_enabled()
        torch.set_grad_enabled(True)
        
        batch_size = self.sample_size
        device = next(self.model.parameters()).device
        s_noise = int(self.sample_size * 0.2)
        s_batch = self.sample_size - s_noise
        inp_imgs = torch.zeros((batch_size, *self.img_shape), device=device)
        inp_imgs[:s_noise] = torch.randn((s_noise, *self.img_shape), device=device)
        if clabel is not None:
            for i in range(s_noise, batch_size):
                c = clabel[i].item()
                if len(self.buffer[c]) > 0:
                    idx = torch.randint(0, len(self.buffer[c]), (1,)).item()
                    inp_imgs[i] = self.buffer[c][idx]
                else:
                    inp_imgs[i] = torch.randn(self.img_shape, device=device)
        else:
            inp_imgs[s_noise:] = torch.randn((s_batch, *self.img_shape), device=device)
        inp_imgs = inp_imgs.detach().requires_grad_(True)
        imgs_per_step = []
        noise = torch.zeros_like(inp_imgs, device=device)
        for _ in range(steps):
            noise.normal_(0, 0.005)
            inp_imgs.data.add_(noise.data)
            inp_imgs.data.clamp_(min=-1.0, max=1.0)
            out_imgs = -self.model(inp_imgs,clabel)
            out_imgs.sum().backward()
            inp_imgs.grad.data.clamp_(-0.03, 0.03)
            inp_imgs.data.add_(-step_size * inp_imgs.grad.data)
            inp_imgs.grad.detach_()
            inp_imgs.grad.zero_()
            inp_imgs.data.clamp_(min=-1.0, max=1.0)
            if return_img_per_step:
                imgs_per_step.append(inp_imgs.clone().detach())
        if clabel is not None:
            for i, c in enumerate(clabel):
                c = c.item()
                if len(self.buffer[c]) < self.cbuffer_size:
                    self.buffer[c] = torch.cat([self.buffer[c], inp_imgs[i].unsqueeze(0)], dim=0)
                else:
                    idx = torch.randint(0, self.cbuffer_size, (1,)).item()
                    self.buffer[c][idx] = inp_imgs[i].detach()
        else:
            for i in range(s_noise, batch_size):
                c = torch.randint(0, self.num_classes,(1,)).item()
                if len(self.buffer[c]) < self.cbuffer_size:
                    self.buffer[c] = torch.cat([self.buffer[c], inp_imgs[i].unsqueeze(0)], dim=0)
                else:
                    idx = torch.randint(0, self.cbuffer_size, (1,)).item()
                    self.buffer[c][idx] = inp_imgs[i].detach()
        for p in self.model.parameters():
            p.requires_grad = True
        self.model.train(is_training)
        torch.set_grad_enabled(had_gradients_enabled)
        return torch.stack(imgs_per_step, dim=0) if return_img_per_step else inp_imgs
    
class JEM(pl.LightningModule):
    def __init__(self, img_shape, batch_size, num_classes=42, cbuffer_size=256, ccond_sample=False, alpha=0.1, lmbd=0.1,
                 lr=1e-4, lr_stepsize=1, lr_gamma=0.97, m_in=0, m_out=-10, steps=60, step_size_decay=1.0, **MODEL_args):
        super().__init__()
        self.save_hyperparameters()
        self.img_shape = img_shape
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.ccond_sample = ccond_sample
        self.cnn = ShallowCNN(**MODEL_args)
        self.lmbd = lmbd
        self.ccond_sample = ccond_sample
        self.steps=steps
        self.alpha =alpha

        self.sampler = MCMCSampler(self.cnn, img_shape=img_shape, sample_size=batch_size, num_classes=num_classes,
                                   cbuffer_size=cbuffer_size)
        self.example_input_array = torch.zeros(1, *img_shape)  

        metrics = torchmetrics.MetricCollection([torchmetrics.CohenKappa(num_classes=num_classes,task='multiclass'),
                                                 torchmetrics.AveragePrecision(num_classes=num_classes,task='multiclass'),
                                                 torchmetrics.AUROC(num_classes=num_classes,task='multiclass'),
                                                 torchmetrics.MatthewsCorrCoef(num_classes=num_classes,task='multiclass'),
                                                 torchmetrics.CalibrationError(task='multiclass',num_classes=num_classes)])
        dyna_metrics = [torchmetrics.Accuracy,
                        torchmetrics.Precision,
                        torchmetrics.Recall,
                        torchmetrics.Specificity,
                        torchmetrics.F1Score]
        self.train_metrics = metrics.clone(prefix='train_')
        self.valid_metrics = metrics.clone(prefix='val_')
        for mode in ['micro', 'macro']:
            self.train_metrics.add_metrics(
                {f"{mode}_{m.__name__}": m(average=mode, num_classes=num_classes,task='multiclass') for m in dyna_metrics})
            self.valid_metrics.add_metrics(
                {f"{mode}_{m.__name__}": m(average=mode, num_classes=num_classes,task='multiclass') for m in dyna_metrics})
        self.hp_metric = torchmetrics.AveragePrecision(num_classes=num_classes,task='multiclass')
        self.valid_metrics = torchmetrics.MetricCollection({
            'multiclass_average_precision': torchmetrics.AveragePrecision(num_classes=num_classes, task='multiclass')
            
        })

    def forward(self, x, labels=None):
        z = self.cnn(x, labels)
        return z
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr, betas=(0.0, 0.999))
        
        scheduler = optim.lr_scheduler.StepLR(optimizer, self.hparams.lr_stepsize,
                                              gamma=self.hparams.lr_gamma)
        return [optimizer], [scheduler]
    
    def px_step(self, batch, ccond_sample=True):      
        px, label = batch
        
        px.add_(px).clamp_(min=-1.0, max=1.0)
        if ccond_sample:
            labels = label
        else:
            labels = None
        px_syn = self.sampler.synthesize_samples(clabel= labels,steps =self.steps)
        inp_imgs = torch.cat([px, px_syn], dim=0)
        real_out, fake_out = self.cnn(inp_imgs).chunk(2, dim=0)
        reg_loss = self.alpha  * (real_out ** 2 + fake_out **2).mean()
        cdiv_loss = fake_out.mean() - real_out.mean()
        loss = reg_loss + cdiv_loss
        return loss
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    def pyx_step(self, batch):
        
        
        px, label = batch
        logits  = self.cnn.get_logits(px)
        loss = F.cross_entropy(logits, label)
        return loss
    def training_step(self, batch, batch_idx):
        
        
        
        
        
        px_loss = self.px_step(batch,ccond_sample=self.ccond_sample)
        pyx_loss = self.pyx_step(batch)
        loss = px_loss + self.lmbd * pyx_loss
        self.log('train_px_loss', px_loss, on_step=True, on_epoch=True)
        self.log('train_pyx_loss', pyx_loss, on_step=True, on_epoch=True)
        self.log('train_loss', loss, on_step=True, on_epoch=True)
        return loss
    def validation_step(self, batch, batch_idx, dataset_idx=None):
        
        
        
        px, label = batch
        logits = self.cnn.get_logits(px)
        loss = F.cross_entropy(logits, label)
        self.valid_metrics.update(logits, label)
        
        
        
        
        
        
        
        
        self.log('val_MulticlassAveragePrecision', self.valid_metrics['multiclass_average_precision'], on_step=False,on_epoch=True)
        return loss
def run_training(args) -> pl.LightningModule:
    """
    Perform EBM/JEM training using a set of hyper-parameters
    Visualization can be either done showcasing different image states during synthesis or by showcasing the
    final results.
    :param args: hyper-parameter
    :return: pl.LightningModule: the trained model
    """
    
    ckpt_dir = args.ckpt_dir
    data_dir = args.data_dir
    num_workers = args.num_workers  
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    num_classes = args.num_classes
    lr = args.lr
    lr_stepsize = args.lr_stepsize
    lr_gamma = args.lr_gamma
    alpha = args.alpha
    cbuffer_size = args.cbuffer_size
    ccond_sample = args.ccond_sample
    
    os.makedirs(ckpt_dir, exist_ok=True)
    
    datasets: Dict[str, TransformTensorDataset] = get_datasets(data_dir)
    train_loader = data.DataLoader(datasets['train'], batch_size=batch_size, shuffle=True, drop_last=True,
                                   num_workers=num_workers, pin_memory=True)
    val_loader = data.DataLoader(datasets['val'], batch_size=batch_size, shuffle=False, drop_last=False,
                                 num_workers=num_workers)
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",  
        mode="min",
        save_top_k=3,
        dirpath=args.ckpt_dir,
        filename="jem-{epoch:02d}-{val_loss:.2f}"
    )
    trainer = pl.Trainer(default_root_dir=ckpt_dir,
                         
                         max_epochs=num_epochs,
                         gradient_clip_val=0.1,
                         callbacks=[
                             
                             
                             ModelCheckpoint(save_weights_only=True, mode="max", monitor='val_MulticlassAveragePrecision',
                                             filename='val_mAP_{epoch}-{step}'),
                             ModelCheckpoint(save_weights_only=True, filename='last_{epoch}-{step}'),
                             LearningRateMonitor("epoch")
                         ])
    pl.seed_everything(42)
    model = JEM(num_epochs=num_epochs,
                img_shape=(1, 56, 56),
                batch_size=batch_size,
                num_classes=num_classes,
                hidden_features=32,  
                cbuffer_size=cbuffer_size,  
                ccond_sample=ccond_sample,  
                lr=lr,  
                lr_gamma=lr_gamma,  
                lr_stepsize=lr_stepsize,  
                alpha=alpha,  
                step_size_decay=1.0  
                )
    trainer.fit(model, train_loader, val_loader)
    trainer.save_checkpoint("/proj/ciptmp/ur03ocab/adl_ex3_files/manual_checkpoint.ckpt")
    model = JEM.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
    return model
def run_generation(args, ckpt_path: Union[str, Path], conditional: bool = False):
    """
    With a trained model we can synthesize new examples from q_\theta using SGLD.
    :param args: hyper-parameter
    :param ckpt_path: local path to the trained checkpoint.
    :param conditional: flag to specify if we want to generate conditioned on a specific class label or not
    :return: None
    """
    model = JEM.load_from_checkpoint(ckpt_path)
    print(model)
    model.to(device)
    pl.seed_everything(42)
    def gen_imgs(model, clabel=None, step_size=10, batch_size=24, num_steps=256):
        model.eval()
        torch.set_grad_enabled(True)  
        mcmc_sampler = MCMCSampler(model, model.img_shape, batch_size, model.num_classes)
        img = mcmc_sampler.synthesize_samples( clabel, steps=num_steps, step_size=step_size, return_img_per_step=True)
        torch.set_grad_enabled(False)
        model.train()
        return img
    k = 8
    bs = 8
    num_steps = 512
    
    conditional_labels = [1, 4, 5, 10, 17, 18, 39, 23]
    synth_imgs = []
    for label in tqdm.tqdm(conditional_labels):
        clabel = (torch.ones(bs) * label).type(torch.LongTensor).to(model.device)
        generated_imgs = gen_imgs(model, clabel=clabel if conditional else None, step_size=10, batch_size=bs, num_steps=num_steps).cpu()
        synth_imgs.append(generated_imgs[-1])
        
        i = 0
        step_size = num_steps // 8
        imgs_to_plot = generated_imgs[step_size - 1::step_size, i]
        imgs_to_plot = torch.cat([generated_imgs[0:1, i], imgs_to_plot], dim=0)
        grid = torchvision.utils.make_grid(imgs_to_plot, nrow=imgs_to_plot.shape[0], normalize=True,
                                           value_range=(-1, 1), pad_value=0.5, padding=2)
        grid = grid.permute(1, 2, 0)
        plt.figure(figsize=(8, 8))
        plt.imshow(grid)
        plt.xlabel("Generation iteration")
        plt.xticks([(generated_imgs.shape[-1] + 2) * (0.5 + j) for j in range(8 + 1)],
                   labels=[1] + list(range(step_size, generated_imgs.shape[0] + 1, step_size)))
        plt.yticks([])
        plt.savefig(f"{'conditional' if conditional else 'unconditional'}_sample_label={label}.png")
    
    grid = torchvision.utils.make_grid(torch.cat(synth_imgs), nrow=k, normalize=True, value_range=(-1, 1),
                                       pad_value=0.5,
                                       padding=2)
    grid = grid.permute(1, 2, 0)
    grid = grid[..., 0].numpy()
    plt.figure(figsize=(12, 24))
    plt.imshow(grid, cmap='Greys')
    plt.xticks([])
    plt.yticks([])
    plt.savefig(f"{'conditional' if conditional else 'unconditional'}_samples.png")
def run_evaluation(args, ckpt_path: Union[str, Path]):
    """
    Evaluate the predictive performance of the JEM model.
    :param args: hyper-parameter
    :param ckpt_path: local path to the trained checkpoint.
    :return: None
    """
    model = JEM.load_from_checkpoint(ckpt_path)
    model.to(device)
    pl.seed_everything(42)
    
    batch_size = args.batch_size
    data_dir = args.data_dir
    num_workers = args.num_workers
    datasets: Dict[str, TransformTensorDataset] = get_datasets(data_dir)
    
    test_loader = data.DataLoader(datasets['test'], batch_size=batch_size, shuffle=False, drop_last=False,
                                  num_workers=num_workers)
    trainer = pl.Trainer() 
    results = trainer.validate(model, dataloaders=test_loader)
    print(results)
    return results
def run_ood_analysis(args, ckpt_path: Union[str, Path]):
    """
    Run out-of-distribution (OOD) analysis. First, you evaluate the scores for the training samples (in-distribution),
    a random noise distribution, and two different distributions that share some resemblence with the training data.
    :param args: hyper-parameter
    :param ckpt_path: local path to the trained checkpoint.
    :return: None
    """
    model = JEM.load_from_checkpoint(ckpt_path)
    model.to(device)
    pl.seed_everything(42)
    
    batch_size = args.batch_size
    data_dir = args.data_dir
    num_workers = args.num_workers
    datasets: Dict[str, TransformTensorDataset] = get_datasets(data_dir)
    
    test_loader = data.DataLoader(datasets['test'], batch_size=batch_size, shuffle=False, drop_last=False,
                                  num_workers=num_workers)
    
    ood_ta_loader = data.DataLoader(datasets['ood_ta'], batch_size=batch_size, shuffle=False, drop_last=False,
                                    num_workers=num_workers)
    ood_tb_loader = data.DataLoader(datasets['ood_tb'], batch_size=batch_size, shuffle=False, drop_last=False,
                                    num_workers=num_workers)
    
    
    
    
    test_scores = score(model, test_loader, score_type="px")
    ood_ta_scores = score(model, ood_ta_loader, score_type="px")
    ood_tb_scores = score(model, ood_tb_loader, score_type="px")
    scores_dict = {"In-Distribution (Test)": test_scores,"OOD Type A": ood_ta_scores,"OOD Type B": ood_tb_scores,}
    plot_histogram(scores_dict, score_type="px")
    auroc_ta, auprc_ta = ood(test_scores, ood_ta_scores)
    auroc_tb, auprc_tb = ood(test_scores, ood_tb_scores)
    print(f"  AUROC: {auroc_ta:.4f}, AUPRC: {auprc_ta:.4f}")
    print(f"  AUROC: {auroc_tb:.4f}, AUPRC: {auprc_tb:.4f}")
def plot_histogram(scores_dict: Dict[str, np.ndarray], score_type: str):
    plt.figure(figsize=(10, 6))
    for label, scores in scores_dict.items():
        plt.hist(scores, bins=50, alpha=0.5, label=label, density=True)
    plt.title(f"Score Distributions ({score_type})")
    plt.xlabel("Score")
    plt.ylabel("Density")
    plt.legend()
    plt.grid()
    plt.show()
def ood(ind_scores, ood_scores):
    y_true = np.concatenate([np.zeros_like(ind_scores), np.ones_like(ood_scores)])
    y_scores = np.concatenate([ind_scores, ood_scores])
    auroc = roc_auc_score(y_true, y_scores)
    auprc = average_precision_score(y_true, y_scores)
    return auroc, auprc
def score(model, loader, score_type="px"):
    scores = []
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(model.device)
            if score_type in ["py", "mass"]
                y = y.to(model.device)
            else:
                y= None
            batch_scores = score_fn(model, x, y, score=score_type)
            scores.append(batch_scores)
    return torch.cat(scores).cpu().numpy()
if __name__ == '__main__':
    args = parse_args()
    
    run_training(args)
    
    ckpt_path: str = "/proj/ciptmp/ur03ocab/adl_ex3_files/manual_checkpoint.ckpt"
    
    run_evaluation(args, ckpt_path)
    
    run_generation(args, ckpt_path, conditional=True)
    run_generation(args, ckpt_path, conditional=False)
    
    run_ood_analysis(args, ckpt_path)
