import os
from dataclasses import asdict
from enum import Enum
from typing import List, Optional, Union
from functools import partial

from datasets import load_from_disk, Dataset
from transformers.utils import logging
from transformers.tokenization_utils import PreTrainedTokenizer

from .core import DataProcessor
from .prep import PREP_PIPELINE
from ...retrieval import SearchBase
from .prep import remove_special_token

logger = logging.get_logger(__name__)


def convert_examples_to_features(
    processor: DataProcessor,
    tokenizer: PreTrainedTokenizer,
    retriever: Optional[SearchBase] = None,
    topk: Optional[int] = 1,
    mode: str = "train",
):
    if mode == "test" and retriever is None:
        raise AttributeError
    
    if mode == "train":
        dataset: Dataset = processor.get_train_examples()
    elif mode == "eval":
        dataset: Dataset = processor.get_eval_examples()
    elif mode == "test":
        dataset: Dataset = processor.get_test_examples()
    else:
        raise NotImplemented
        
    logger.info(f"[{mode.upper()}] convert examples to features")

    prep_pipeline = PREP_PIPELINE[processor.model_args.reader_type]
    
    prep_fn, is_batched = prep_pipeline(tokenizer, mode, processor.data_args)
    
    if retriever is not None:
        eval_mode = mode == "eval"
        dataset = retriever.retrieve(dataset, topk=topk, eval_mode=eval_mode)
        prep_fn = partial(prep_fn, retriever=retriever)
    
    features = dataset.map(
        prep_fn,
        batched=is_batched,
        num_proc=processor.data_args.preprocessing_num_workers,
        remove_columns=dataset.column_names,
        load_from_cache_file=not processor.data_args.overwrite_cache,
    )

    if ('v3' in processor.data_args.dataset_version) & (retriever is None) & (mode != "test"):
        dataset = dataset.map(
            remove_special_token,
            batched=is_batched,
            num_proc=processor.data_args.preprocessing_num_workers,
            load_from_cache_file=not processor.data_args.overwrite_cache,
        )
            
    return features, dataset
    
    
class OdqaProcessor(DataProcessor):
    
    def get_train_examples(self):
        dataset_path = self.data_args.dataset_path
        if self.data_args.curriculum_learning:
            input_data = load_from_disk(os.path.join(dataset_path, "train_dataset"))[self.data_args.curriculum_split_name]
        else:
            input_data = load_from_disk(os.path.join(dataset_path, "train_dataset"))["train_date_mask"]
        return input_data
    
    def get_eval_examples(self):
        dataset_path = self.data_args.dataset_path
        input_data = load_from_disk(os.path.join(dataset_path, "train_dataset"))["validation"]
        return input_data
    
    def get_test_examples(self):
        dataset_path = self.data_args.dataset_path
        input_data = load_from_disk(os.path.join(dataset_path, "test_dataset"))["validation"]
        return input_data