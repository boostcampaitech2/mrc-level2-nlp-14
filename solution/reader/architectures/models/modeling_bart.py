from transformers import (
    BartForQuestionAnswering as QA,
    BartForConditionalGeneration as CG,
    BartPretrainedModel,
    BartModel,
)

from ..modeling_heads import QAConvSDSHead


class BartForQuestionAnswering(QA):
    reader_type: str = "extractive"
    
    def __init__(self, config):
        super().__init__(config)
        assert config.reader_type == self.reader_type
        
        
class BartForConditionalGeneration(CG):
    reader_type: str = "generative"
    
    def __init__(self, config):
        super().__init__(config)
        assert config.reader_type == self.reader_type


class BartForQAWithConvSDSHead(QA):
    reader_type: str = "extractive"
    
    def __init__(self, config):
        # Initialize on BartPretrainedModel
        super(BartForQuestionAnswering, self).__init__(config)
        assert config.reader_type == self.reader_type
        
        config.num_labels = 2
        self.num_labels = config.num_labels
        
        self.model = BartModel(config)
        self.qa_outputs = QAConvSDSHead(
            config.qa_conv_input_size,
            config.hidden_size,
            config.qa_conv_n_layers,
            config.num_labels,
        )
        
        self.model._init_weights(self.qa_outputs)