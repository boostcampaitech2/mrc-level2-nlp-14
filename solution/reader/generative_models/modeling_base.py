
from transformers import AutoModelForSeq2SeqLM
from solution.reader.core import ReaderModelBase


class GenerativeReaderModel(AutoModelForSeq2SeqLM, ReaderModelBase):
    """
    HF Model with a generation head on top for generative question-answering tasks like SQuAD (a linear
    layer on top of the hidden-states output to generate `span start logits` and `span end logits`).
    """