from transformers import Trainer, Seq2SeqTrainer

from .mixin import ToMixin


class BaseTrainer(Trainer, ToMixin):
    pass


class Seq2SeqBaseTrainer(Seq2SeqTrainer, ToMixin):
    pass