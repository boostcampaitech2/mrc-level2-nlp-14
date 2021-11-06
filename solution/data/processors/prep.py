from solution.utils.constant import (
    QUESTION_COLUMN_NAME,
    CONTEXT_COLUMN_NAME,
    ANSWER_COLUMN_NAME,
)

from .corrupt import permute_sentences
import re

DENOISE_FUNC = {
    "sentence_permutation": permute_sentences,
}

def remove_special_token(examples):
    """
    Remove special tokens in data v3

    Args:
        examples (Dict[Any]): DatasetDict

    Returns:
        Dict[Any]: DatasetDict
    """
    answers = []
    context = []
    document_id = []
    ids = []
    question = []
    title = []

    for i in range(len(examples['context'])):    
        sentence_list = examples['context'][i].split('#')
        result = sentence_list.copy()

        result = ' '.join(sentence_list)
        index = result.find('[ANSWER]')
        result = re.sub('\[ANSWER\]','',result)

        answer = examples['answers'][i]
        answer['answer_start'][0] = index

        answers.append(answer)
        context.append(result)
        document_id.append(examples['document_id'][i])
        ids.append(examples['id'][i])
        question.append(examples['question'][i])
        title.append(examples['title'][i])

    return {'answers': answers,
            'context': context,
            'document_id': document_id,
            'id': ids,
            'question': question,
            'title': title}

def get_extractive_features(tokenizer, mode, data_args):
    
    def tokenize_fn(examples):
        pad_on_right = tokenizer.padding_side == "right"
        max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

        # truncation과 padding을 통해 tokenization을 진행
        # stride를 이용하여 overflow를 유지
        # 각 example들은 이전의 context와 조금씩 겹침
        # overflow 발생 시 지정한 batch size보다 더 많은 sample이 들어올 수 있음 -> data augmentation
        tokenized_examples = tokenizer(
            examples[QUESTION_COLUMN_NAME if pad_on_right else CONTEXT_COLUMN_NAME],
            examples[CONTEXT_COLUMN_NAME if pad_on_right else QUESTION_COLUMN_NAME],
            # 길이가 긴 context가 등장할 경우 truncation을 진행
            truncation="only_second" if pad_on_right else "only_first",
            max_length=max_seq_length,
            stride=data_args.doc_stride,
            # overflow 발생 시 원래 인덱스를 찾을 수 있게 mapping 가능한 값이 필요
            return_overflowing_tokens=True,
            # token의 캐릭터 단위 position을 찾을 수 있는 offset을 반환
            # start position과 end position을 찾는데 도움을 줌
            return_offsets_mapping=True,
            # sentence pair가 입력으로 들어올 때 0과 1로 구분지음
            return_token_type_ids=data_args.return_token_type_ids,
            padding="max_length" if data_args.pad_to_max_length else False,
        )
        return tokenized_examples

    def prepare_train_features(examples):
        pad_on_right = tokenizer.padding_side == "right"
        # denoising
        if 'v3' in data_args.dataset_version:
            if data_args.denoising_func:
                examples = DENOISE_FUNC[data_args.denoising_func](examples, data_args)
            else:
                examples = remove_special_token(examples)

        tokenized_examples = tokenize_fn(examples)

        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        offset_mapping = tokenized_examples.pop("offset_mapping")

        # 데이터셋에 "start position", "enc position" label을 부여합니다.
        tokenized_examples["start_positions"] = []
        tokenized_examples["end_positions"] = []

        for i, offsets in enumerate(offset_mapping):
            input_ids = tokenized_examples["input_ids"][i]
            cls_index = input_ids.index(tokenizer.cls_token_id)  # cls index

            # sequence id를 설정합니다 (context와 question을 구분).
            sequence_ids = tokenized_examples.sequence_ids(i)
            context_index = 1 if pad_on_right else 0

            # 길이가 긴 context에 대해 truncation을 진행하기 때문에
            # 하나의 example이 여러 개의 span을 가질 수 있음
            sample_index = sample_mapping[i]
            answers = examples[ANSWER_COLUMN_NAME][sample_index]

            # answer가 없을 경우 cls_index를 answer로 설정
            # example에서 정답이 없는 경우가 있을 수 있음
            if len(answers["answer_start"]) == 0:
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # text에서 정답의 start/end character index를 가져옴
                start_char = answers["answer_start"][0]
                end_char = start_char + len(answers["text"][0])

                # sequence_ids는 0, 1, None의 세 값만 가짐
                # None 0 0 ... 0 None 1 1 ... 1 None

                # text에서 context가 시작하는 위치로 이동
                token_start_index = 0
                while sequence_ids[token_start_index] != context_index:
                    token_start_index += 1

                # text에서 context가 끝나는 위치로 이동
                token_end_index = len(input_ids) - 1
                while sequence_ids[token_end_index] != context_index:
                    token_end_index -= 1

                # 정답이 span을 벗어나는지 체크.
                # 정답이 없는 경우 CLS index로 labeling (Retro일 경우 다르게 처리)
                if not (
                    offsets[token_start_index][0] <= start_char and
                    offsets[token_end_index][1] >= end_char
                ):
                    tokenized_examples["start_positions"].append(cls_index)
                    tokenized_examples["end_positions"].append(cls_index)
                else:
                    # token_start_index 및 token_end_index를 answer의 끝으로 이동
                    # Note: answer가 마지막 단어인 경우 last offset을 따라갈 수 있음

                    # token_start_index를 실제 위치로 맞춰주는 과정
                    while (
                        token_start_index < len(offsets) and 
                        offsets[token_start_index][0] <= start_char
                    ):
                        token_start_index += 1
                    tokenized_examples["start_positions"].append(token_start_index - 1)

                    # token_end_index를 실제 위치로 맞춰주는 과정
                    while offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    tokenized_examples["end_positions"].append(token_end_index + 1)

        return tokenized_examples
    
    def prepare_validation_features(examples, retriever=None):
        pad_on_right = tokenizer.padding_side == "right"

        # odqa.py 에서 v3 설명 참조. dataset version이 v3.*.*이고 retrieval 하지 않을 때 실행
        if ('v3' in data_args.dataset_version) & (retriever is None):
            examples = remove_special_token(examples)

        tokenized_examples = tokenize_fn(examples)

        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

        # evaluation을 위해 prediction을 context의 substring으로 변환
        # corresponding example_id를 유지하고 offset mappings을 저장
        tokenized_examples["example_id"] = []

        for i in range(len(tokenized_examples["input_ids"])):
            # sequence id를 설정합니다 (context와 question을 구분).
            sequence_ids = tokenized_examples.sequence_ids(i)
            context_index = 1 if pad_on_right else 0

            # 하나의 example이 여러 개의 span을 가질 수 있음
            sample_index = sample_mapping[i]
            tokenized_examples["example_id"].append(examples["id"][sample_index])

            tokenized_examples["offset_mapping"][i] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(tokenized_examples["offset_mapping"][i])
            ]

        return tokenized_examples
    
    def identity(examples):
        return examples
    
    if mode == "train":
        get_features_fn = prepare_train_features
    elif mode == "eval":
        get_features_fn = prepare_validation_features
    elif mode == "test":
        get_features_fn = prepare_validation_features
    
    return get_features_fn, True


def get_generative_features(tokenizer, mode, data_args):

    def tokenize_fn(examples):
        model_inputs = [f"질문: {q} 지문: {c} </s>"
                        for q, c in zip(examples["question"], examples["context"])]
        output = tokenizer(
            model_inputs,
            max_length=data_args.max_seq_length,
            padding=data_args.pad_to_max_length,
            truncation=True,
        )
        output.update({"example_id": [e_id for e_id in examples["id"]]})
        return output
    
    def tokenize_fn_labels(examples):
        labels = [f"{answer['text'][0]} </s>" for answer in examples["answers"]]
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                labels,
                max_length=data_args.max_label_length,
                padding=data_args.pad_to_max_length,
                truncation=True,
            )["input_ids"]
        return labels
    
    def prepare_train_features(examples):
        tokenized_examples = tokenize_fn(examples)
        labels = tokenize_fn_labels(examples)
        tokenized_examples.update({"labels": labels})
        return tokenized_examples
    
    def prepare_test_features(examples):
        tokenized_examples = tokenize_fn(examples)
        return tokenized_examples
    
    if mode == "train":
        get_features_fn = prepare_train_features
    elif mode == "eval":
        get_features_fn = prepare_train_features
    elif mode == "test":
        get_features_fn = prepare_test_features

    return get_features_fn, True


def get_ensemble_features(tokenizer, mode, data_args):

    def tokenize_fn(examples):
        output = tokenizer(
            [f"<s> 질문: {q} 지문: </s>" for q in examples["question"]],
            [f"{c} </s>" for c in examples["context"]],
            max_length=data_args.max_seq_length,
            padding=data_args.pad_to_max_length,
            return_offsets_mapping=True,
            truncation=True,
        )
        output.update({"example_id": [e_id for e_id in examples["id"]]})
        return output

    def tokenize_fn_labels(examples):
        labels = [f"{answer['text'][0]} </s>" for answer in examples["answers"]]
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                labels,
                max_length=data_args.max_label_length,
                padding=data_args.pad_to_max_length,
                truncation=True,
            )["input_ids"]
        return labels

    def prepare_train_features(examples):
        pad_on_right = tokenizer.padding_side == "right"
        tokenized_examples = tokenize_fn(examples)
        labels = tokenize_fn_labels(examples)
        tokenized_examples.update({"labels": labels})

        offset_mapping = tokenized_examples.pop("offset_mapping")

        tokenized_examples["start_positions"] = []
        tokenized_examples["end_positions"] = []

        for i, offsets in enumerate(offset_mapping):
            input_ids = tokenized_examples["input_ids"][i]
            cls_index = input_ids.index(tokenizer.cls_token_id)  # cls index
            sep_index = input_ids.index(tokenizer.sep_token_id) # sep index

            offsets[cls_index] = (0, 0) # 필수!
            offsets[sep_index] = (0, 0) # 필수!

            sequence_ids = tokenized_examples.sequence_ids(i)
            answers = examples[ANSWER_COLUMN_NAME][i]
            context_index = 0 if pad_on_right else 1

            if len(answers["answer_start"]) == 0:
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                start_char = answers["answer_start"][0]
                end_char = start_char + len(answers["text"][0])
                    
                token_start_index = 0
                while sequence_ids[token_start_index] == context_index:
                    token_start_index += 1

                token_end_index = len(input_ids) - 1
                while sequence_ids[token_end_index] == context_index or input_ids[token_end_index] == tokenizer.pad_token_id:
                    token_end_index -= 1

                if not (
                    offsets[token_start_index][0] <= start_char and
                    offsets[token_end_index][1] >= end_char
                ):
                    tokenized_examples["start_positions"].append(cls_index)
                    tokenized_examples["end_positions"].append(cls_index)
                else:
                    while (
                        token_start_index < len(offsets) and 
                        offsets[token_start_index][0] <= start_char
                    ):
                        token_start_index += 1
                    tokenized_examples["start_positions"].append(token_start_index - 1)

                    while offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    tokenized_examples["end_positions"].append(token_end_index + 1)

        return tokenized_examples

    
    def prepare_test_features(examples):
        pad_on_right = tokenizer.padding_side == "right"
        tokenized_examples = tokenize_fn(examples)
        tokenized_examples["example_id"] = []

        for i, input_ids in enumerate(tokenized_examples["input_ids"]):
            sequence_ids = tokenized_examples.sequence_ids(i)
            context_index = 0 if pad_on_right else 1
            tokenized_examples["example_id"].append(examples["id"][i])
            tokenized_examples["offset_mapping"][i] = [
                (o if sequence_ids[k] != context_index and input_ids[k] != tokenizer.pad_token_id else None)
                for k, o in enumerate(tokenized_examples["offset_mapping"][i])
            ]

        return tokenized_examples


    if mode == "train":
        get_features_fn = prepare_train_features
    elif mode == "eval":
        get_features_fn = prepare_train_features
    elif mode == "test":
        get_features_fn = prepare_test_features

    return get_features_fn, True


PREP_PIPELINE = {
    "extractive": get_extractive_features,
    "generative": get_generative_features,
    "ensemble" : get_ensemble_features,
}