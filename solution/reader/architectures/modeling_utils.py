import math
import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F


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
    
    
class AttentionLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.query_layer = nn.Linear(50*config.hidden_size, config.hidden_size, bias=True)
        self.key_layer = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.value_layer = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, token_type_ids):
        embedded_query = x * (token_type_ids.unsqueeze(dim=-1)==0) # 전체 Text 중 query에 해당하는 Embedded Vector만 남김.
        embedded_query = self.dropout(F.relu(embedded_query)) # Activation Function 및 Dropout Layer 통과
        embedded_query = embedded_query[:, :50, :] # 질문에 해당하는 Embedding만 남김. (B * 50 * hidden_size)
        embedded_query = embedded_query.reshape((x.shape[0], 1, -1)) # Token의 Embedding을 Hidden Dim축으로 Concat함. (B * 1 * 50 x hidden_size)
        embedded_query = self.query_layer(embedded_query) # Dense Layer를 통과 시킴. (B * 1 * hidden_size)
        
        embedded_key = x * (token_type_ids.unsqueeze(dim=-1)==1) # 전체 Text 중 context에 해당하는 Embedded Vector만 남김.
        embedded_key = self.key_layer(embedded_key) # (B * max_seq_length * hidden_size)
        
        attention_rate = torch.matmul(embedded_key, torch.transpose(embedded_query, 1, 2)) # Context의 Value Vector와 Quetion의 Query Vector를 사용 (B * max_seq_legth * 1)
        attention_rate = attention_rate / math.sqrt(embedded_key.shape[-1]) # hidden size의 표준편차로 나눠줌. (B * max_seq_legth * 1)
        attention_rate = attention_rate / 10 # Temperature로 나눠줌. (B * max_seq_legth * 1)
        attention_rate = F.softmax(attention_rate, 1) # softmax를 통과시켜서 확률값으로 변경해, Question과 Context의 Attention Rate를 구함. (B * max_seq_legth * 1)
        
        embedded_value = self.value_layer(x) # (B * max_seq_length * hidden_size)
        embedded_value = embedded_value * attention_rate # Attention Rate를 활용해서 Output 값을 변경함. (B * max_seq_legth * hidden_size)
        return embedded_value
    
    
class ConvLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels=config.hidden_size, 
            out_channels=config.qa_conv_out_channel, 
            kernel_size=1)

        self.conv3 = nn.Conv1d(
            in_channels=config.hidden_size, 
            out_channels=config.qa_conv_out_channel, 
            kernel_size=3,
            padding=1)

        self.conv5 = nn.Conv1d(
            in_channels=config.hidden_size, 
            out_channels=config.qa_conv_out_channel, 
            kernel_size=5, 
            padding=2)

        self.drop_out = nn.Dropout(0.3)

    def forward(self, x):
        conv_input = x.transpose(1, 2) # Convolution 연산을 위해 Transpose (B * hidden_size * max_seq_legth)
        conv_output1 = F.relu(self.conv1(conv_input)) # Conv연산의 결과 (B * num_conv_filter * max_seq_legth)
        conv_output3 = F.relu(self.conv3(conv_input)) # Conv연산의 결과 (B * num_conv_filter * max_seq_legth)
        conv_output5 = F.relu(self.conv5(conv_input)) # Conv연산의 결과 (B * num_conv_filter * max_seq_legth)
        concat_output = torch.cat((conv_output1, conv_output3, conv_output5), dim=1) # Concatenation (B * num_conv_filter x 3 * max_seq_legth)

        concat_output = concat_output.transpose(1, 2) # Dense Layer에 입력을 위해 Transpose (B * max_seq_legth * num_conv_filter x 3)
        concat_output = self.drop_out(concat_output) # dropout 통과
        return concat_output