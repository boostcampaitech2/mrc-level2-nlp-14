import torch
import torch.nn as nn
    
    
class QAConvSDSHead(nn.Module):
    
    def __init__(
        self,
        input_size: int,
        hidden_dim: int,
        n_layers: int,
        num_labels: int,
    ):
        super().__init__()
        convs = []
        for n in range(n_layers):
            convs.append(QAConvSDSLayer(input_size, hidden_dim))
        self.convs = nn.Sequential(*convs)
        self.qa_output = nn.Linear(hidden_dim, num_labels)
    
    def forward(self, x):
        out = self.convs(x)
        return self.qa_output(out)