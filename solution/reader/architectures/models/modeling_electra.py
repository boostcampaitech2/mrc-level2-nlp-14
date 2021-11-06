from transformers import (
    ElectraForQuestionAnswering as QA,
    ElectraPreTrainedModel,
    ElectraModel,
)

from ..modeling_heads import QAConvSDSHead


class ElectraForQuestionAnswering(QA):
    reader_type: str = "extractive"

    def __init__(self, config):
        super().__init__(config)
        assert config.reader_type == self.reader_type


class ElectraForQAWithConvSDSHead(QA):
    reader_type: str = "extractive"

    def __init__(self, config):
        # Initialize on ElectraPreTrainedModel
        super(ElectraForQuestionAnswering, self).__init__(config)
        assert config.reader_type == self.reader_type

        config.num_labels = 2
        self.num_labels = config.num_labels

        self.electra = ElectraModel(config)
        self.qa_outputs = QAConvSDSHead(
            config.qa_conv_input_size,
            config.hidden_size,
            config.qa_conv_n_layers,
            config.num_labels,
        )

        self.init_weights()
