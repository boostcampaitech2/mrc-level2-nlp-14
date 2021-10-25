from transformers import (
    RobertaForQuestionAnswering,
    RobertaPreTrainedModel,
    RobertaModel,
)

from ..modeling_heads import QAConvSDSHead


class RobertaForQA(RobertaForQuestionAnswering):
    reader_type: str = "extractive"
    
    def __init__(self, config):
        super().__init__(config)
        assert config.reader_type == self.reader_type


class RobertaForQAWithConvSDSHead(RobertaForQuestionAnswering):
    reader_type: str = "extractive"
    
    def __init__(self, config):
        # Initialize on RobertaPreTrainedModel
        super(RobertaForQuestionAnswering, self).__init__(config)
        assert config.reader_type == self.reader_type
        
        config.num_labels = 2
        self.num_labels = config.num_labels
        
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.qa_outputs = QAConvSDSHead(
            config.qa_conv_input_size,
            config.hidden_size,
            config.qa_conv_n_layers,
            config.num_labels,
        )
        
        self.init_weights()