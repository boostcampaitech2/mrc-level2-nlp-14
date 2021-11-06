from transformers import PreTrainedModel

from .core import ReaderBase
from .architectures import (
    AutoModelForQuestionAnswering,
    AutoModelForSeq2SeqLM,
)
from .trainers import (
    BaseTrainer,
    QuestionAnsweringTrainer,
    QuestionAnsweringSeq2SeqTrainer,
    QuestionAnsweringEnsembleTrainer,
)


class ExtractiveReader(ReaderBase):
    reader_type: str = "extractive"
    default_model: PreTrainedModel = AutoModelForQuestionAnswering
    default_trainer: BaseTrainer = QuestionAnsweringTrainer


class GenerativeReader(ReaderBase):
    reader_type: str = "generative"
    default_model: PreTrainedModel = AutoModelForSeq2SeqLM
    default_trainer: BaseTrainer = QuestionAnsweringSeq2SeqTrainer


class EnsembleReader(ReaderBase):
    reader_type: str = "ensemble"
    default_model: PreTrainedModel = AutoModelForSeq2SeqLM
    default_trainer: BaseTrainer = QuestionAnsweringEnsembleTrainer


class RetroReader(ReaderBase):
    reader_type: str = "retrospective"
    default_model: PreTrainedModel = None
    default_trainer: BaseTrainer = None
