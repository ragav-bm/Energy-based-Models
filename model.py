import torch
import torch.nn as nn
import torch.nn.functional as F

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)
    
class ShallowCNN(nn.Module):
    def __init__(self, hidden_features=32, num_classes=42, **kwargs):
        super().__init__()
        c_hid1 = hidden_features
        c_hid2 = hidden_features * 2
        c_hid3 = hidden_features * 4
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(1, c_hid1, kernel_size=5, stride=2, padding=4),
            Swish(),
            nn.Conv2d(c_hid1, c_hid2, kernel_size=3, stride=2, padding=1),
            Swish(),
            nn.Conv2d(c_hid2, c_hid3, kernel_size=3, stride=2, padding=1),
            Swish(),
            nn.Conv2d(c_hid3, c_hid3, kernel_size=3, stride=2, padding=1),
            Swish(),
            nn.Conv2d(c_hid3, c_hid3, kernel_size=3, stride=2, padding=1),
            Swish(),
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c_hid3, num_classes)
        )

    def get_logits(self, x):
        
        logits1 = self.cnn_layers(x)
        logits_1D = F.adaptive_avg_pool2d(logits1,output_size=(1,1))
        # logits_1D = logits_1D.view(logits_1D.shape[0], -1)
        logits = self.fc_layers(logits_1D)
        return logits
    
    def forward(self, x, y=None) -> torch.Tensor:
        logits = self.get_logits(x)
        if y is None:
            return torch.logsumexp(logits, dim=1)
        else:
            return logits.gather(1,y.view(-1,1)).squeeze()
