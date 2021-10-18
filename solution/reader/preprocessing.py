import os
from functools import partial

from transformers import (
    HfArgumentParser,
    TrainingArguments,
)
import transformers

from solution.args import project_args

from solution.args import (
    HfArgumentParser,
    get_args_parser,
    DataArguments,
    ModelingArguments,
    NewTrainingArguments,
    ProjectArguments
)
from .constant import (
    QUESTION_COLUMN_NAME,
    CONTEXT_COLUMN_NAME,
    ANSWER_COLUMN_NAME,
)

def tokenize_examples(examples, tokenizer):
    command_args = get_args_parser()
    parser = HfArgumentParser(
        (DataArguments, NewTrainingArguments, ModelingArguments, ProjectArguments)
    )
    data_args, _, _, _ = parser.parse_yaml_file(yaml_file=os.path.abspath(command_args.config))

    pad_on_right = tokenizer.padding_side == "right"
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    # truncation과 padding(length가 짧을때만)을 통해 toknization을 진행하며, stride를 이용하여 overflow를 유지합니다.
    # 각 example들은 이전의 context와 조금씩 겹치게됩니다.
    tokenized_examples = tokenizer(
        examples[QUESTION_COLUMN_NAME if pad_on_right else CONTEXT_COLUMN_NAME],
        examples[CONTEXT_COLUMN_NAME if pad_on_right else QUESTION_COLUMN_NAME],
        truncation="only_second" if pad_on_right else "only_first",
        max_length=max_seq_length,
        stride=data_args.doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        #return_token_type_ids=False, # roberta모델을 사용할 경우 False, bert를 사용할 경우 True로 표기해야합니다.
        padding="max_length" if data_args.pad_to_max_length else False,
    )
    return tokenized_examples


# Train preprocessing / 전처리를 진행합니다.
def prepare_train_features(examples, tokenizer):
    tokenized_examples = tokenize_examples(examples, tokenizer)
    pad_on_right = tokenizer.padding_side == "right"

    # 길이가 긴 context가 등장할 경우 truncate를 진행해야하므로, 해당 데이터셋을 찾을 수 있도록 mapping 가능한 값이 필요합니다.
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    # token의 캐릭터 단위 position를 찾을 수 있도록 offset mapping을 사용합니다.
    # start_positions과 end_positions을 찾는데 도움을 줄 수 있습니다.
    offset_mapping = tokenized_examples.pop("offset_mapping")

    # 데이터셋에 "start position", "enc position" label을 부여합니다.
    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"] = []

    for i, offsets in enumerate(offset_mapping):
        input_ids = tokenized_examples["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)  # cls index

        # sequence id를 설정합니다 (to know what is the context and what is the question).
        sequence_ids = tokenized_examples.sequence_ids(i)

        # 하나의 example이 여러개의 span을 가질 수 있습니다.
        sample_index = sample_mapping[i]
        answers = examples[ANSWER_COLUMN_NAME][sample_index]

        # answer가 없을 경우 cls_index를 answer로 설정합니다(== example에서 정답이 없는 경우 존재할 수 있음).
        if len(answers["answer_start"]) == 0:
            tokenized_examples["start_positions"].append(cls_index)
            tokenized_examples["end_positions"].append(cls_index)
        else:
            # text에서 정답의 Start/end character index
            start_char = answers["answer_start"][0]
            end_char = start_char + len(answers["text"][0])

            # text에서 current span의 Start token index
            token_start_index = 0
            while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
                token_start_index += 1

            # text에서 current span의 End token index
            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
                token_end_index -= 1

            # 정답이 span을 벗어났는지 확인합니다(정답이 없는 경우 CLS index로 label되어있음).
            if not (
                offsets[token_start_index][0] <= start_char
                and offsets[token_end_index][1] >= end_char
            ):
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # token_start_index 및 token_end_index를 answer의 끝으로 이동합니다.
                # Note: answer가 마지막 단어인 경우 last offset을 따라갈 수 있습니다(edge case).
                while (
                    token_start_index < len(offsets)
                    and offsets[token_start_index][0] <= start_char
                ):
                    token_start_index += 1
                tokenized_examples["start_positions"].append(token_start_index - 1)
                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                tokenized_examples["end_positions"].append(token_end_index + 1)
    return tokenized_examples


# Validation preprocessing
def prepare_validation_features(examples, tokenizer):
    tokenized_examples = tokenize_examples(examples, tokenizer)
    pad_on_right = tokenizer.padding_side == "right"

    # 길이가 긴 context가 등장할 경우 truncate를 진행해야하므로, 해당 데이터셋을 찾을 수 있도록 mapping 가능한 값이 필요합니다.
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

    # evaluation을 위해, prediction을 context의 substring으로 변환해야합니다.
    # corresponding example_id를 유지하고 offset mappings을 저장해야합니다.
    tokenized_examples["example_id"] = []

    for i in range(len(tokenized_examples["input_ids"])):
        # sequence id를 설정합니다 (to know what is the context and what is the question).
        sequence_ids = tokenized_examples.sequence_ids(i)
        context_index = 1 if pad_on_right else 0

        # 하나의 example이 여러개의 span을 가질 수 있습니다.
        sample_index = sample_mapping[i]
        tokenized_examples["example_id"].append(examples["id"][sample_index])

        # Set to None the offset_mapping을 None으로 설정해서 token position이 context의 일부인지 쉽게 판별 할 수 있습니다.
        tokenized_examples["offset_mapping"][i] = [
            (o if sequence_ids[k] == context_index else None)
            for k, o in enumerate(tokenized_examples["offset_mapping"][i])
        ]
    return tokenized_examples

PREPARE_FEATURES = {'train' : prepare_train_features,
                    'valid' : prepare_validation_features}

def prepare_features(split:str, tokenizer:transformers.PreTrainedTokenizer):
    """ Get prepare functions. prepare_train_features or prepare_validation_features """
    return partial(
                PREPARE_FEATURES[split],
                tokenizer=tokenizer,
            )