import inspect
from typing import Optional, Callable, List, Any
from functools import partial
from contextlib import contextmanager

import torch
from datasets import Dataset

from transformers.utils import logging
from transformers import PreTrainedTokenizer

from .trainers import BaseTrainer
from ..args import ModelArguments
from .architectures import MODEL_INIT


logger = logging.get_logger(__name__)

TRAINER_BASE_PARAMS = list(inspect.signature(BaseTrainer.__init__).parameters)


class ReaderBase:
    """
    Reader base class used for MRC task.
    Every reader object has a main method :func:`read`.
    Just like humans, the read method of the Reader class reads (train), evaluates it,
    and makes predictions (inferences) based on what it reads.
    The above function is implemented as train, evaluate, and predict functions of
    huggingface trainer, and input parameters and functions are defined with the `mode_change` method.
    """

    _mode: str = "train"
    mode_candidate: List[str] = ["train", "evaluate", "predict"]

    def __init__(
        self,
        model_args: ModelArguments,
        tokenizer: PreTrainedTokenizer,
    ):
        """
        When a Reader object is created, the model argument and tokenzier are used to
        define the model initialization function to be used for training or inference.
        """
        self._trainer = None
        self.model_args = model_args
        model_init = self.get_model_init_func(model_args.model_init)
        self.model_init = partial(
            model_init,
            model_args=model_args,
            default_model=self.default_model,
            tokenizer=tokenizer,
        )

    def get_model_init_func(self, model_init: str) -> Callable:
        """Get the hf model initialization function to be used for training or inference.

        Args:
            model_init (str): Name of the model initialization function to use.

        Returns:
            Callable: The function to be used to initialize the model.
        """

        model_init = MODEL_INIT.get(model_init, None)
        if model_init is None:
            raise AttributeError("Requires model initialization function.")
        return model_init

    @property
    def trainer_params(self) -> List[str]:
        """
        Returns the parameters required
        when creating the default trainer of the reader object.
        """
        trainer_init = self.default_trainer.__init__
        signature = inspect.signature(trainer_init)
        return list(signature.parameters)

    def set_trainer(self, **kwargs):
        """
        Set up the hf trainer to be used for training, evaluation and inference.
        The trainer is a class provided by 🤗
        and creates an object using the signature of the trainer that inherited it.
        """
        params = {}
        for key in kwargs:
            if key in TRAINER_BASE_PARAMS:
                params.update({key: kwargs.get(key)})
            elif key in self.trainer_params:
                params.update({key: kwargs.get(key)})
            else:
                raise AttributeError(
                    "You entered an argument that does not match the initialization parameter "
                    f"of {self.default_trainer} registered as the default trainer "
                    f"of the class {self.__class__.__name__}."
                    "Please check your signature used for trainer initialization."
                )
        self._trainer = self.default_trainer(**params)

    @property
    def mode(self) -> str:
        """ Return current mode. """
        return self._mode

    @mode.setter
    def mode(self, val: str):
        """ Set the mode with input value. """
        assert val in self.mode_candidate, (
            f"You only assign {self.mode_candidate} to mode."
        )
        self._mode = val

    @contextmanager
    def mode_change(self, mode: str):
        """
        Change read mode using contextmanager.
        When exiting the decorator, it restores its original mode.
        """
        _mode = self.mode
        assert mode in self.mode_candidate, (
            f"You only assign {self.mode_candidate} to mode."
        )
        self.mode = mode
        yield
        self.mode = _mode

    def num_examples(self, dataset: Dataset) -> int:
        """Return the number of examples of dataset.

        Args:
            dataset (Dataset)

        Returns:
            [int]: number of examples of dataset.
        """

        return len(dataset)

    def save_trainer(self):
        """ Save trainer's model and states on trainer.args.outpur_dir path. """
        logger.warning("Save trainer states, model, and tokenizer.")
        self._trainer.save_model()
        self._trainer.save_state()

    def save_metrics(self, split: str, metrics: Any, dataset=None, combined=True):
        """ Saves the result metric of trainer training or evaluation. """

        logger.info("Save metrics")
        if dataset is not None:
            metrics[f"{split}_samples"] = self.num_examples(dataset)
        self._trainer.log_metrics(split, metrics)
        self._trainer.save_metrics(split, metrics, combined=True)

    def read(self, *args, **kwargs) -> Any:
        """
        Main method for Reader class.

        Use the Trainer class from huggingface transformers.
        The methods supported by this class are train, evaluate, and test.
        It is recommended to change the mode with `mode_change` contextmanager
        because it determines which method to use as the mode attribute of the Reader instance.

        If the trainer to be used for reading is not defined or
        if arguments other than keyword arguments are entered, an error is returned.
        """

        assert self._trainer is not None, (
            "There is no trainer class that is the subject to use the read method. "
            "Please define a trainer using the `set_trainer` method."
        )
        assert not args, (
            "The input of the method receives only key word arguments unconditionally. "
            "Check the signatures of `train`, `evaluate`, and `predict` of the Trainer class of 🤗 transformers "
            "and put the appropriate input into kwargs."
        )
        logger.warning(f"***** {self.mode.title()} *****")
        trainer_method = getattr(self._trainer, self.mode)
        params = inspect.signature(trainer_method).parameters
        use_kwargs = {k: kwargs[k] for k in kwargs.keys() if k in params}
        results = trainer_method(**use_kwargs)
        return results
