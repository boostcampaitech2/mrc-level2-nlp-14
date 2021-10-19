# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import abc

from dataclasses import asdict, dataclass

from datasets import load_from_disk, Dataset
from numpy.core.numeric import NaN
from transformers import AutoTokenizer
from functools import partial

from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    set_seed,
)

from solution.args import (
    DataArguments,
    NewTrainingArguments,
    ModelingArguments,
)

from solution.utils import (
    check_no_error,
)

from solution.reader.extractive_models import EXT_MODEL_INIT_FUNC
from solution.reader.generative_models import GEN_MODEL_INIT_FUNC


class ReaderBase():
    """ Base class for Reader module """
    
    def __init__(self, data_args, training_args, model_args):
        @dataclass
        class Args:
            model_args: ModelingArguments
            data_args: DataArguments
            training_args: NewTrainingArguments

        self.args = Args(model_args=model_args,
                        data_args=data_args,
                        training_args=training_args)
    
    @abc.abstractmethod
    def data_collator(self):
        """ Get dataset (fix name convention) """
        pass

    @abc.abstractmethod
    def datasets(self):
        """ Get train_dataset (fix name convention) """
        pass

    @abc.abstractmethod
    def train_dataset(self):
        """ Get train_dataset (fix name convention) """
        pass

    @abc.abstractmethod
    def eval_dataset(self):
        """ Get eval_dataset (fix name convention) """
        pass

    @abc.abstractmethod
    def retrieved_eval_dataset(self):
        """ Get eval_dataset (fix name convention) """
        pass

    @abc.abstractmethod
    def post_process_function(self):
        """ Get post_process_function (fix name convention) """
        pass

    @abc.abstractmethod
    def max_seq_length(self):
        """ Get post_process_function (fix name convention) """
        pass

    @abc.abstractmethod
    def model_config(self):
        """ Get model_config (fix name convention) """
        pass

    @abc.abstractmethod
    def model_init(self):
        """ Get model init function (fix name convention) """
        pass

    @abc.abstractmethod
    def tokenizer(self):
        """ Get tokenizer (fix name convention) """
        pass

    @abc.abstractmethod
    def pre_process_function(self):
        """ Get pre_process_function (fix name convention) """
        pass

    @abc.abstractmethod
    def trainer(self):
        """ Get trainer (fix name convention) """
        pass

    @abc.abstractmethod
    def last_checkpoint(self):
        """ Get post_process_function (fix name convention) """
        pass

    

    def _set_initial_setup(self):
        """ Initial Set up attributes """
        # Seed를 고정하고 전체 데이터셋과 train, test set을 불러옵니다.
        set_seed(self.args.training_args.seed)
        self.datasets = load_from_disk(self.args.data_args.dataset_name)
        root_data_dir = os.path.dirname(self.args.data_args.dataset_name)
        self.test_datasets = load_from_disk(os.path.join(root_data_dir, 'test_dataset'))

        # AutoConfig를 이용하여 pretrained model 과 tokenizer를 불러옵니다.
        # argument로 원하는 모델 이름을 설정하면 옵션을 바꿀 수 있습니다.
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.args.model_args.tokenizer_name
            if self.args.model_args.tokenizer_name
            else self.args.model_args.model_name_or_path,
            # 'use_fast' argument를 True로 설정할 경우 rust로 구현된 tokenizer를 사용할 수 있습니다.
            # False로 설정할 경우 python으로 구현된 tokenizer를 사용할 수 있으며,
            # rust version이 비교적 속도가 빠릅니다.
            use_fast=True,
        )
        
        if self.args.model_args.method == "ext":
            _model_init = EXT_MODEL_INIT_FUNC.get(self.args.model_args.model_init)
        elif self.args.model_args.method == "gen":
            _model_init = GEN_MODEL_INIT_FUNC.get(self.args.model_args.model_init)

        if _model_init is None:
            raise ValueError("Check whether architecture is properly set or not")

        self.model_init = partial(_model_init,
                            model_args=self.args.model_args,
                            )



        # Data collator
        # flag가 True이면 이미 max length로 padding된 상태입니다.
        # 그렇지 않다면 data collator에서 padding을 진행해야합니다.
        self.data_collator = DataCollatorWithPadding(
            self.tokenizer, pad_to_multiple_of=8 if self.args.training_args.fp16 else None
        )

        self.logger.info(
            type(self.args.training_args),
            type(self.args.model_args),
            type(self.datasets),
            type(self.tokenizer),
            type(self.model_init),
        )
        self.logger.info("*** Initial set-up of the Reader Model Completed ***")

    def _set_preprocessing(self):
        """ Pre-process the datasets """
        # dataset을 전처리합니다.
        # training과 evaluation에서 사용되는 전처리는 아주 조금 다른 형태를 가집니다.
        if self.args.training_args.do_train:
            column_names = self.datasets["train"].column_names
        else:
            column_names = self.datasets["validation"].column_names

        # 오류가 있는지 확인합니다.
        self.last_checkpoint, max_seq_length = check_no_error(
            self.args.data_args, self.args.training_args, self.datasets, self.tokenizer
        )
        
        if self.args.training_args.do_train:
            if "train" not in self.datasets:
                raise ValueError("--do_train requires a train dataset")
            self.train_dataset = self.datasets["train"]

            # dataset에서 train feature를 생성합니다.
            self.train_dataset = self.train_dataset.map(
                self.pre_process_function(split='train', tokenizer=self.tokenizer),
                batched=True,
                num_proc=self.args.data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not self.args.data_args.overwrite_cache,
            )

        if self.args.training_args.do_eval:
            if "train" not in self.datasets:
                raise ValueError("--do_train requires a train dataset")
            self.eval_dataset = self.datasets["validation"]

            # Validation Feature 생성
            self.eval_dataset = self.eval_dataset.map(
                self.pre_process_function('valid', tokenizer=self.tokenizer),
                batched=True,
                num_proc=self.args.data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not self.args.data_args.overwrite_cache,
            )

        # train_dataset에서 불러온 validation -> context 존재 or retrieval로 대체
        # test_dataset -> validation -> context 없음, only retrieval

        self.logger.info("*** Pre-process the Datasets Completed ***")

    def preprocessing_retrieved_doc(self, retrieved_examples:Dataset):
        """ Pre-process the retrieved validation datasets """
        # Context가 Retrieved passage로 채워진 validation set에 대해 전처리를 수행합니다.
        column_names = retrieved_examples.column_names
        retrieved_dataset = retrieved_examples.map(
            self.pre_process_function('valid', tokenizer=self.tokenizer),
            batched=True,
            num_proc=self.args.data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not self.args.data_args.overwrite_cache,
        )
        self.logger.info("*** Pre-process the Retrieved Dataset Completed ***")
        return retrieved_dataset

    @abc.abstractmethod
    def set_trainer(self):
        """ Set Hugginface Trainer """
        pass

    @abc.abstractmethod
    def train(self, *args, **kwargs):
        """ Call train method of self.trainer """
        pass

    @abc.abstractmethod
    def evaluate(self, *args, **kwargs):
        """ Call evaluate method of self.trainer """
        pass

    @abc.abstractmethod
    def predict(self, *args, **kwargs):
        """ Call predict method of self.trainer """
        pass
