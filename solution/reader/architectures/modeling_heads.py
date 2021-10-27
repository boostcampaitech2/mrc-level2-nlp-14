import torch
import torch.nn as nn

from .modeling_utils import (
    QAConvSDSLayer,
    ConvLayer,
    AttentionLayer
)
    
    
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
    
    
class QAConvHeadWithAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = AttentionLayer(config)
        self.conv = ConvLayer(config)
        self.classify_layer = nn.Linear(
            config.qa_conv_out_channel*3, 2, bias=True)

    def forward(self, x, token_type_ids):
        embedded_value = self.attention(x, token_type_ids)
        concat_output = self.conv(embedded_value)
        logits = self.classify_layer(concat_output)
        return logits


class QAConvHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.conv = ConvLayer(config)
        self.classify_layer = nn.Linear(
            config.qa_conv_out_channel*3, 2, bias=True)

    def forward(self, **kwargs):#x, input_ids, sep_token_id):
        concat_output = self.conv(kwargs['x'])
        logits = self.classify_layer(concat_output)
        return logits