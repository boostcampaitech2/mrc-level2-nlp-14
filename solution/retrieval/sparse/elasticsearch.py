from solution.models.modling_convolution import QAConvLayer

class QAConvLayer(nn.Module):
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: List[int],
        padding: int,
        n_layers: int,
    ):
        for layer in range(n_layer):
            for k in kernel_size:
                setattr(self, f"conv_{layer}_{i}", 
                        nn.Conv1d(in_channels=in_channels, 
                                  out_channels=out_channels, 
                                  kernel_size=k, 
                                  padding=padding)
                       )

class MyReaderModel(~~):
    
    def __init__(self, config):
        super().__init__(config)
        
        self.backbone = AutoModel.from_pretrained(~)
        self.QAConvLayer = QAConvLayer(config)