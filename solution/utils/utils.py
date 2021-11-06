import os
import gc
import time
import psutil
import numpy as np
import random
import torch

from typing import Tuple, Any, Callable, List, Union, Dict
from transformers import is_torch_available

from transformers import PreTrainedTokenizerFast

from transformers.utils import logging
from transformers.trainer_utils import get_last_checkpoint

from datasets import Dataset, DatasetDict

from solution.args import (
    MrcDataArguments,
    MrcTrainingArguments,
)


logger = logging.get_logger(__name__)


def _check_usage_of_cpu_and_memory():
    pid = os.getpid()
    py = psutil.Process(pid)
    cpu_usage = os.popen("ps aux | grep " + str(pid) +
                         " | grep -v grep | awk '{print $3}'").read()
    cpu_usage = cpu_usage.replace("\n", "")
    memory_usage = round(py.memory_info()[0] / 2.**30, 2)
    print("cpu usage\t\t:", cpu_usage, "%")
    print("memory usage\t\t:", memory_usage, "%")
    print("cuda memory allocated\t:", torch.cuda.memory_allocated())
    print("cuda memory reserved\t:", torch.cuda.memory_reserved())


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
    """Seed fixer (random, numpy, torch)

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
    data_args: MrcDataArguments,
    training_args: MrcTrainingArguments,
    tokenizer,
) -> Tuple[Any, int]:

    # Find last checkpoint
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
            logger.warning(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Tokenizer check: this script needs Fast tokenizer
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

    return last_checkpoint, max_seq_length
