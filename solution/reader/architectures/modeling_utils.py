import torch
import torch.nn as nn


class QAConvSDSLayer(nn.Module):
    
    def __init__(
        self,
        input_size: int,
        hidden_dim: int,
    ):
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels=input_size, 
            out_channels=input_size*2, 
            kernel_size=3, 
            padding=1
        )
        self.conv2 = nn.Conv1d(
            in_channels=input_size*2, 
            out_channels=input_size, 
            kernel_size=1, 
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = x + torch.relu(out)
        out = self.layer_norm(out)
        return out