from .modeling_bart import *
from .modeling_bert import *
from .modeling_electra import *
from .modeling_mt5 import *
from .modeling_roberta import *


from transformers import (
    AutoModelForQuestionAnswering as AutoQA,
    AutoModelForSeq2SeqLM as AutoS2SLM,
)


class AutoModelForQuestionAnswering(AutoQA):
    reader_type: str = "extractive"

    def __init__(self, config):
        super().__init__(config)
        assert config.reader_type == self.reader_type


class AutoModelForSeq2SeqLM(AutoS2SLM):
    reader_type: str = "generative"

    def __init__(self, config):
        super().__init__(config)
        assert config.reader_type == self.reader_type
