class MCMCSampler:
    def __init__(self, model, img_shape, sample_size, num_classes, cbuffer_size=256):
        self.model = model
        self.img_shape = img_shape
        self.sample_size = sample_size
        self.num_classes = num_classes
        self.cbuffer_size = cbuffer_size
        self.cbuffer = [[] for _ in range(num_classes)] if num_classes else []

    def synthesize_samples(self, clabel=None, steps=60, step_size=15, return_img_per_step=False):
        is_training = self.model.training
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        had_gradients_enabled = torch.is_grad_enabled()
        torch.set_grad_enabled(True)
        s_noise = int(self.sample_size * 0.2)
        s_batch = self.sample_size - s_noise
        
        dataset_noise = torch.randn(s_noise, *self.img_shape) * 0.01
                
        clabel = (s_batch,clabel)
        print("ffffffffffffffffffffffffffffffffffffffffffffffffffffff")
        print(clabel)
        
        dataset_batch = self._sample_from_buffer(clabel, s_batch) if clabel is not None else torch.randn(s_batch, *self.img_shape) * 0.01
        
        inp_imgs = torch.cat([dataset_batch, dataset_noise], dim=0).to(next(self.model.parameters()).device)
        inp_imgs.requires_grad = True
        
        noise = torch.randn(inp_imgs.shape, device=inp_imgs.device)
        
        imgs_per_step = []
        
        for step in range(steps):
            noise.normal_(0, 0.005)
            inp_imgs.data.add_(noise.data)
            inp_imgs.data.clamp_(min=-1.0, max=1.0)  
            
            out_imgs = -self.model(inp_imgs)
            out_imgs.sum().backward()
            inp_imgs.grad.data.clamp_(-0.03, 0.03)  
            
            inp_imgs.data.add_(-step_size * inp_imgs.grad.data)
            inp_imgs.grad.detach_()
            inp_imgs.grad.zero_()
            inp_imgs.data.clamp_(min=-1.0, max=1.0)
            if return_img_per_step:
                imgs_per_step.append(inp_imgs.clone().detach())
            
        sysn_dataset = inp_imgs.detach().cpu()
        
        if clabel is not None:
            self._update_reservoir(self.cbuffer[clabel], sysn_dataset)
        else:
            for i in range(self.num_classes):
                self._update_reservoir(self.cbuffer[i], sysn_dataset[i::self.num_classes])
        
        for p in self.model.parameters():
            p.requires_grad = True
        self.model.train(is_training)
        torch.set_grad_enabled(had_gradients_enabled)
        return torch.stack(imgs_per_step, dim=0) if return_img_per_step else inp_imgs
    
    def _sample_from_buffer(self, clabel, s_batch):

        buffer_sample = self.cbuffer[clabel]
        if len(buffer_sample) == 0:
            return torch.randn(s_batch, *self.img_shape) * 0.01
        if len(buffer_sample) < s_batch:
            noise_padding = torch.randn(s_batch - len(buffer_sample), *self.img_shape) * 0.01
            return torch.cat([torch.stack(buffer_sample), noise_padding], dim=0)
        indices = torch.randint(0, len(buffer_sample), (s_batch,))
        selected_samples = [buffer_sample[i] for i in indices]
        return torch.stack(selected_samples, dim=0)
    
    def _update_reservoir(self, cbuffer, sysn_dataset):
        for s in sysn_dataset:
            if len(cbuffer) < self.cbuffer_size:
                cbuffer.append(s)
            else:
                replace_idx = random.randint(0, len(cbuffer)-1)
                cbuffer[replace_idx] = s