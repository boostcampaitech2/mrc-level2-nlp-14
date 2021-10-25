from transformers import (
    BertForQuestionAnswering as QA,
    BertPreTrainedModel,
    BertModel,
)

from ..modeling_heads import QAConvSDSHead


class BertForQuestionAnswering(QA):
    reader_type: str = "extractive"
    
    def __init__(self, config):
        super().__init__(config)
        assert config.reader_type == self.reader_type


class BertForQAWithConvSDSHead(QA):
    reader_type: str = "extractive"
    
    def __init__(self, config):
        # Initialize on BertPreTrainedModel
        super(BertForQuestionAnswering, self).__init__(config)
        assert config.reader_type == self.reader_type
        
        config.num_labels = 2
        self.num_labels = config.num_labels
        
        self.bert = BertModel(config, add_pooling_layer=False)
        self.qa_outputs = QAConvSDSHead(
            config.qa_conv_input_size,
            config.hidden_size,
            config.qa_conv_n_layers,
            config.num_labels,
        )
        
        self.init_weights()