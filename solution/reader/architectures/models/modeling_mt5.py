from transformers import (
    MT5ForConditionalGeneration as CG,
)


class MT5ForConditionalGeneration(CG):
    reader_type: str = "generative"
    
    def __init__(self, config):
        super().__init__(config)
        assert config.reader_type == self.reader_type