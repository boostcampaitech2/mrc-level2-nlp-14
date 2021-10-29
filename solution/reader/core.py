import inspect
from typing import Optional, Callable, List
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
    _mode: str = "train"
    mode_candidate: List[str] = ["train", "evaluate", "predict"]
    
    def __init__(
        self, 
        model_args: ModelArguments, 
        tokenizer: PreTrainedTokenizer,
    ):
        self._trainer = None
        self.model_args = model_args
        model_init = self.get_model_init_func(model_args.model_init)
        self.model_init = partial(
            model_init,
            model_args=model_args,
            default_model=self.default_model,
            tokenizer=tokenizer,
        )
        
    def get_model_init_func(self, model_init: str):
        model_init = MODEL_INIT.get(model_init, None)
        if model_init is None:
            raise AttributeError
        return model_init
    
    @property
    def trainer_params(self):
        trainer_init = self.default_trainer.__init__
        signature = inspect.signature(trainer_init)
        return list(signature.parameters)
    
    def set_trainer(self, **kwargs):
        params = {}
        for key in kwargs:
            if key in TRAINER_BASE_PARAMS:
                params.update({key: kwargs.get(key)})
            elif key in self.trainer_params:
                params.update({key: kwargs.get(key)})
            else:
                raise AttributeError
        #assert (params.get("tokenizer", None) is None or
        #        params.get("data_collator", None) is None)
        self._trainer = self.default_trainer(**params)
        
    @property
    def mode(self):
        return self._mode
    
    @mode.setter
    def mode(self, val: str):
        assert val in self.mode_candidate
        self._mode = val
        
    @contextmanager
    def mode_change(self, mode: str):
        _mode = self.mode
        assert mode in self.mode_candidate
        self.mode = mode
        yield
        self.mode = _mode
        
    def num_examples(self, dataset: Dataset):
        return len(dataset)
        
    def save_trainer(self):
        logger.warning("Save trainer states, model, and tokenizer.")
        self._trainer.save_model()
        self._trainer.save_state()
        
    def save_metrics(self, split, metrics, dataset=None, combined=True):
        logger.info("Save metrics")
        if dataset is not None:
            metrics[f"{split}_samples"] = self.num_examples(dataset)
        self._trainer.log_metrics(split, metrics)
        self._trainer.save_metrics(split, metrics, combined=True)
    
    def read(self, *args, **kwargs):
        assert self._trainer is not None
        assert not args
        logger.warning(f"***** {self.mode.title()} *****")
        trainer_method = getattr(self._trainer, self.mode)
        params = inspect.signature(trainer_method).parameters
        use_kwargs = {k:kwargs[k] for k in kwargs.keys() if k in params}
        results = trainer_method(**use_kwargs)
        return results