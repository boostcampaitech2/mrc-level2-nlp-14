import os
import numpy as np
import random
import torch
import logging

from typing import Tuple, Any, Callable, List, Union, Dict
from transformers import is_torch_available

from transformers import PreTrainedTokenizerFast

from ..args import NewTrainingArguments
from transformers.trainer_utils import get_last_checkpoint

from datasets import Dataset, DatasetDict

from solution.args import (
    DataArguments,
)


logger = logging.getLogger(__name__)


def timer(dataset: bool = True):
    def decorator(func: Callable):
        """ Time decorator """
        flag = True if dataset else False
        def wrap_func(self, query_or_dataset: Union[str, Dataset], *args, **kwargs):
            dataset_cond = isinstance(query_or_dataset, Dataset) and flag
            str_cond = isinstance(query_or_dataset, str) and not flag
            if dataset_cond or str_cond:
                t0 = time.time()
            output = func(self, query_or_dataset, *args, **kwargs)
            if dataset_cond or str_cond:
                print(f"[{func.__name__}] done in {time.time() - t0:.3f} s")
            return output
        return wrap_func
    return decorator


def set_seed(seed: int = 42):
    """
    seed 고정하는 함수 (random, numpy, torch)

    Args:
        seed (:obj:`int`): The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    if is_torch_available():
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if use multi-GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def check_no_error(
    data_args: DataArguments,
    training_args: NewTrainingArguments,
    datasets: DatasetDict,
    tokenizer,
) -> Tuple[Any, int]:

    # last checkpoint 찾기.
    last_checkpoint = None
    if (
        os.path.isdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Tokenizer check: 해당 script는 Fast tokenizer를 필요로합니다.
    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        raise ValueError(
            "This example script only works for models that have a fast tokenizer. Checkout the big table of models "
            "at https://huggingface.co/transformers/index.html#bigtable to find the model types that meet this "
            "requirement"
        )

    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warn(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    if "validation" not in datasets:
        raise ValueError("--do_eval requires a validation dataset")
    return last_checkpoint, max_seq_length