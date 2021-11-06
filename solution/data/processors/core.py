from dataclasses import dataclass, field
from typing import List, Optional, Union

from ...args import (
    DataArguments,
    ModelArguments,
    TrainingArguments,
)


class DataProcessor:
    """ Base class for data converters """

    def __init__(
        self,
        data_args: DataArguments,
        model_args: ModelArguments,
        training_args: TrainingArguments,
    ):
        self.data_args = data_args
        self.model_args = model_args
        self.training_args = training_args

    def get_train_examples(self, data_dir):
        """Gets a collection of :class:`InputExample` for the train set."""
        raise NotImplementedError()

    def get_eval_examples(self, data_dir):
        """Gets a collection of :class:`InputExample` for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of :class:`InputExample` for the test set."""
        raise NotImplementedError()
