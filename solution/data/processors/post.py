# coding=utf-8
# Copyright 2020 The HuggingFace Team All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import collections
import json
import os
from collections import Counter
from typing import Optional, Tuple

import numpy as np
from tqdm.auto import tqdm

from transformers import EvalPrediction
from transformers.utils import logging
from solution.utils.constant import ANSWER_COLUMN_NAME

from konlpy.tag import Mecab, Okt, Kkma, Komoran
from khaiii import KhaiiiApi


logger = logging.get_logger(__name__)


def make_bracket_pair(
    text
):
    """
    If the beginning or end of the text is in brackets, but there is only one side, match the pair,
    and if '(' appears in the middle of the text but ')' does not appear at the end, remove the the back of '(' and return it

    Args:
        text ([str]): pred_answer.

    Returns:
        [str]: post processed text with bracket pair
    """

    pair_punc_1 = '〈〉≪≫《》「」『』‘’“”'
    pair_punc_2 = '<>＜＞'
    none_pair_punc = '"\''
    tup_punc_1 = tuple(pair_punc_1)
    tup_punc_2 = tuple(pair_punc_2)
    tup_none_punc = tuple(none_pair_punc)

    # startswith
    if text.startswith(tup_punc_1) and chr(ord(text[0])+1) not in text:
        text += chr(ord(text[0])+1)

    if text.startswith(tup_punc_2) and chr(ord(text[0])+2) not in text:
        text += chr(ord(text[0])+2)

    if text.startswith(tup_none_punc) and text.count(text[0]) == 1:
        text += text[0]

    # endswith
    if text.endswith(tup_punc_1) and chr(ord(text[-1])-1) not in text:
        text = chr(ord(text[-1])-1) + text

    if text.endswith(tup_punc_2) and chr(ord(text[-1])-2) not in text:
        text = chr(ord(text[-1])-2) + text

    if text.endswith(tup_none_punc) and text.count(text[-1]) == 1:
        text = text[-1] + text

    # deletion
    if not text.startswith('(') and '(' in text and ')' not in text:
        text = text[:text.find('(')]

    return text


def get_pos_tagged_from_word(
    pred_answer, analyzer
):
    """
    Progress morpheme analysis.

    Args:
        pred_answer ([str]): predicted answer.
        analyzer ([type]): part-of-speech tagger(kaiii, mecab, okt, kkma, komoran).

    Returns:
        List[Tuple(str)]: the result of pred_answer's morpheme analysis.
    """

    pos_tagged_answer = analyzer.pos(pred_answer)

    return pos_tagged_answer


def get_pos_tagged_from_sentence(
    ref_text, stride, pred_answer, analyzer
):
    """
    The part corresponding to pred_answer among the morpheme analysis results of ref_text is returned,
    and if an IndexError occurs during the mapping process, it is returned as get_pos_tagged_from_word.

    Args:
        ref_text ([str]): text obtained by more stride based on pred_answer in context.
        stride ([int]): based on pred_answer, how many chars will be fetched from side to side in the context.
        pred_answer ([str]): predicted answer.
        analyzer ([type]): part-of-speech tagger(kaiii, mecab, okt, kkma, komoran).

    Returns:
        List[Tuple(str)]: the result of pred_answer's morpheme analysis.
    """

    ref_text_pos = analyzer.pos(ref_text)
    ref_text_reverse = list(ref_text)[::-1]
    ref_to_pos_idx = []

    for i, m in enumerate(ref_text_pos):
        if ref_text_reverse[-1:] == [' ']:
            ref_text_reverse.pop()
            ref_to_pos_idx.append('_')

        for j in range(len(m[0])):
            try:
                ref_text_reverse.pop()
                ref_to_pos_idx.append(i)
            except IndexError:
                return get_pos_tagged_from_word(pred_answer, analyzer)

        if ref_text_reverse[-1:] == [' ']:
            ref_text_reverse.pop()
            ref_to_pos_idx.append('_')

    if stride == 0:
        target = ref_to_pos_idx[:]
    else:
        target = ref_to_pos_idx[stride:-stride]

    if target == []:
        return get_pos_tagged_from_word(pred_answer, analyzer)

    try:
        if ref_text_pos[target[0]] != '_' and ref_to_pos_idx[target[-1]+1] != '_':
            pos_tagged_answer = ref_text_pos[target[0]:target[-1]+1]
        elif ref_text_pos[target[0]] == '_' and ref_to_pos_idx[(target[-1])+1] != '_':
            pos_tagged_answer = ref_text_pos[target[0]-1:target[-1]+1]
        else:
            pos_tagged_answer = get_pos_tagged_from_word(pred_answer, analyzer)

    except IndexError:
        pos_tagged_answer = get_pos_tagged_from_word(pred_answer, analyzer)

    return pos_tagged_answer


def get_pos_ensemble(
    pred_answer, ref_text, stride
):
    """
    After determining whether pred_answer ends with an postposition based on the results of the morpheme analysis ensemble,
    if it ends with an postposition, remove the postposition and return it.

    Args:
        pred_answer ([str]): predicted answer.
        ref_text ([str]): text obtained by more stride based on pred_answer in context.
        stride ([int]): based on pred_answer, how many chars will be fetched from side to side in the context.

    Returns:
        [str]: if the end of the pred_answer is an postposition, it will be removed.
    """

    kaiii = KhaiiiApi()
    mecab = Mecab()
    okt = Okt()
    kkma = Kkma()
    komoran = Komoran()

    kaiii_tagged_anser = [(morph.lex, morph.tag) for word in kaiii.analyze(
        pred_answer) for morph in word.morphs]
    mecab_tagged_answer = get_pos_tagged_from_sentence(
        ref_text, stride, pred_answer, mecab)
    okt_tagged_answer = get_pos_tagged_from_sentence(
        ref_text, stride, pred_answer, okt)
    kkma_tagged_answer = get_pos_tagged_from_sentence(
        ref_text, stride, pred_answer, kkma)
    komoran_tagged_answer = get_pos_tagged_from_sentence(
        ref_text, stride, pred_answer, komoran)

    pos_tagged_answer = {
        'kaiii': kaiii_tagged_anser,
        'mecab': mecab_tagged_answer,
        'okt': okt_tagged_answer,
        'kkma': kkma_tagged_answer,
        'komoran': komoran_tagged_answer,
    }

    postposition_list = [pos_tag[-1][0] for key,
                         pos_tag in pos_tagged_answer.items() if pos_tag[-1][1].startswith("J")]

    if len(postposition_list) >= 4:
        remove_len = len(sorted(Counter(postposition_list).items(),
                         key=lambda x: x[1], reverse=True)[0][0])
        pred_answer = pred_answer[:-remove_len]

    return pred_answer


def pred_answer_post_process(
    context, offsets
):
    """
    After preprocessing of pred_answer as the main function of the post-processing function,
    return the result through get_pos_enssemble, make_bracket_pair.

    Args:
        context ([str]): context referenced to get pred_answer.
        offsets (List[int]): index for finding a location in context based on tokenizer index.

    Returns:
        [str]: post-processed text.
    """

    pred_answer = context[offsets[0]: offsets[1]]

    if pred_answer.startswith(' '):
        offsets[0] += 1
    if pred_answer.endswith(' '):
        offsets[1] -= 1

    pred_answer = pred_answer.strip()

    stride = 15
    ref_text = context[(offsets[0])-stride:(offsets[1])+stride]
    while (offsets[0])-stride < 0 or (offsets[1])+stride > len(context):
        stride -= 1
        ref_text = context[(offsets[0])-stride:(offsets[1])+stride]

    removal_tag_list = ['[TITLE]', '[ANSWER]']
    for tag in removal_tag_list:
        if tag in pred_answer:
            processed_answer = pred_answer.split(tag)[-1]
            removed_answer = pred_answer.split(tag)[0]
            offsets[0] += (len(removed_answer) + len(tag))
            ref_text = ref_text[:ref_text.find(
                removed_answer)] + ref_text[(ref_text.find(processed_answer)):]
            pred_answer = processed_answer

    if pred_answer.startswith('#'):
        offsets[0] += 1
        pred_answer = pred_answer[1:]
        ref_text = ref_text[:ref_text.find(
            '#')] + ref_text[(ref_text.find('#'))+1:]

    if pred_answer.endswith('#'):
        offsets[1] -= 1
        pred_answer = pred_answer[:-1]
        ref_text = ref_text[:ref_text.rfind(
            '#'):] + ref_text[(ref_text.rfind('#'))+1:]

    post_ensembled_answer = get_pos_ensemble(pred_answer, ref_text, stride)
    pred_result = make_bracket_pair(post_ensembled_answer)

    return pred_result


def save_pred_json(
    all_predictions, all_nbest_json, output_dir, prefix
):
    """
    Save prediction.json, nbest_predctions.json in output_dir.

    Args:
        all_predictions ([Dict]): total prediction to be updated.
        all_nbest_json ([Dict]): total prediction of nbest size to be updated.
        output_dir ([str]): output directory.
        prefix ([str]): prefix to distinguish data to be stored.
    """

    assert os.path.isdir(output_dir), f"{output_dir} is not a directory."

    prediction_file = os.path.join(
        output_dir,
        "predictions.json" if prefix is None else f"predictions_{prefix}.json",
    )
    nbest_file = os.path.join(
        output_dir,
        "nbest_predictions.json"
        if prefix is None
        else f"nbest_predictions_{prefix}.json",
    )

    logger.info(f"Saving predictions to {prediction_file}.")
    with open(prediction_file, "w", encoding="utf-8") as writer:
        writer.write(
            json.dumps(all_predictions, indent=4, ensure_ascii=False) + "\n"
        )
    logger.info(f"Saving nbest_preds to {nbest_file}.")
    with open(nbest_file, "w", encoding="utf-8") as writer:
        writer.write(
            json.dumps(all_nbest_json, indent=4, ensure_ascii=False) + "\n"
        )


def get_all_logits(
    predictions, features
):
    """
    After checking assertions against predictions and features length,
    return start & end logtis.

    Args:
        predictions ([Tuple[ndarray, ndarray]]): start & end logit predictions.
        features ([Dataset]): tokenized & splited datasets.

    Returns:
        Tuple([array]): start & end logtis
    """

    assert (
        len(predictions) == 2
    ), "`predictions` should be a tuple with two elements (start_logits, end_logits)."
    all_start_logits, all_end_logits = predictions

    assert len(predictions[0]) == len(
        features
    ), f"Got {len(predictions[0])} predictions and {len(features)} features."

    return all_start_logits, all_end_logits


def map_features_to_example(
    examples, features
):
    """
    Returns the mapping of feature indices to example index.

    Args:
        examples ([Dataset]): raw datasets.
        features ([Dataset]): tokenized & splited datasets.

    Returns:
        [Dict[List]]: the mapping of feature indices to example index.
    """

    # example과 mapping되는 feature 생성
    example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
    features_per_example = collections.defaultdict(list)
    for i, feature in enumerate(features):
        features_per_example[example_id_to_index[feature["example_id"]]].append(
            i)

    return features_per_example


def get_candidate_preds(
    features, feature_indices, all_start_logits, all_end_logits, n_best_size, max_answer_length
):
    """
    It returns predictions of n_best_size of features mapped to one exmaple.

    Args:
        features ([Dataset]): tokenized & splited datasets.
        feature_indices ([List]): feature indices of one loop exmaple.
        all_start_logits ([ndarray]): all start logits.
        all_end_logits ([ndarray]): all end logits.
        n_best_size ([int]): number of return best predictions.
        max_answer_length ([int]): max span of answer.

    Returns:
        [List[Dict]]: predictions of n_best_size of features mapped to one exmaple.
    """

    min_null_prediction = None
    prelim_predictions = []

    for feature_index in feature_indices:
        # 각 featureure에 대한 모든 prediction을 가져옵니다.
        start_logits = all_start_logits[feature_index]
        end_logits = all_end_logits[feature_index]
        # logit과 original context의 logit을 mapping합니다.
        offset_mapping = features[feature_index]["offset_mapping"]
        # Optional : `token_is_max_context`, 제공되는 경우 현재 기능에서 사용할 수 있는 max context가 없는 answer를 제거합니다
        token_is_max_context = features[feature_index].get(
            "token_is_max_context", None
        )

        # minimum null prediction을 업데이트 합니다.
        feature_null_score = start_logits[0] + end_logits[0]
        if (
            min_null_prediction is None
            or min_null_prediction["score"] > feature_null_score
        ):
            min_null_prediction = {
                "offsets": (0, 0),
                "score": feature_null_score,
                "start_logit": start_logits[0],
                "end_logit": end_logits[0],
            }

        # `n_best_size`보다 큰 start and end logits을 살펴봅니다.
        start_indexes = np.argsort(start_logits)[
            -1: -n_best_size - 1: -1
        ].tolist()

        end_indexes = np.argsort(end_logits)[
            -1: -n_best_size - 1: -1
        ].tolist()

        for start_index in start_indexes:
            for end_index in end_indexes:
                # out-of-scope answers는 고려하지 않습니다.
                if (
                    start_index >= len(offset_mapping)
                    or end_index >= len(offset_mapping)
                    or offset_mapping[start_index] is None
                    or offset_mapping[end_index] is None
                ):
                    continue
                # 길이가 < 0 또는 > max_answer_length인 answer도 고려하지 않습니다.
                if (
                    end_index < start_index
                    or end_index - start_index + 1 > max_answer_length
                ):
                    continue
                # 최대 context가 없는 answer도 고려하지 않습니다.
                if (
                    token_is_max_context is not None
                    and not token_is_max_context.get(str(start_index), False)
                ):
                    continue
                prelim_predictions.append(
                    {
                        "offsets": (
                            offset_mapping[start_index][0],
                            offset_mapping[end_index][1],
                        ),
                        "score": start_logits[start_index] + end_logits[end_index],
                        "start_logit": start_logits[start_index],
                        "end_logit": end_logits[end_index],
                    }
                )

    # 가장 좋은 `n_best_size` predictions만 유지합니다.
    predictions = sorted(
        prelim_predictions, key=lambda x: x["score"], reverse=True
    )[:n_best_size]

    return predictions


def get_example_prediction(
    example, predictions, all_predictions, all_nbest_json, do_pos_ensemble
):
    """
    Convert offset from a presentation from an excel to an answer text,
    and return by adding a presentation to all_prediction and all_nbest_json.

    Args:
        example ([Dataset]): raw datasets.
        predictions ([List[Dict]]): prediction of one example.
        all_predictions ([Dict]): total prediction to be updated.
        all_nbest_json ([Dict]): total prediction of nbest size to be updated.

    Returns:
        [List[Dict]]:
    """
    # predict text offset mapping & post-processed pred_answer
    context = example["context"]
    for pred in predictions:
        offsets = pred.pop("offsets")
        pred["text"] = context[offsets[0]: offsets[1]]
        pred["offsets"] = list(offsets)

    # rare edge case에는 null이 아닌 예측이 하나도 없으며 failure를 피하기 위해 fake prediction을 만듭니다.
    if len(predictions) == 0 or (
        len(predictions) == 1 and predictions[0]["text"] == ""
    ):

        predictions.insert(
            0, {"text": "empty", "start_logit": 0.0,
                "end_logit": 0.0, "score": 0.0}
        )

    # 모든 점수의 소프트맥스를 계산합니다(we do it with numpy to stay independent from torch/tf in this file, using the LogSumExp trick).
    scores = np.array([pred.pop("score") for pred in predictions])
    exp_scores = np.exp(scores - np.max(scores))
    probs = exp_scores / exp_scores.sum()

    # 예측값에 확률을 포함합니다.
    for prob, pred in zip(probs, predictions):
        pred["probability"] = prob

    # predict일 경우 진행
    if do_pos_ensemble:
        predictions[0]["text"] = pred_answer_post_process(
            context, predictions[0]["offsets"])

    # best prediction을 선택합니다.
    all_predictions[example["id"]] = predictions[0]["text"]

    # np.float를 다시 float로 casting -> `predictions`은 JSON-serializable 가능
    all_nbest_json[example["id"]] = [
        {
            k: (
                float(v)
                if isinstance(v, (np.float16, np.float32, np.float64))
                else v
            )
            for k, v in pred.items()
        }
        for pred in predictions
    ]

    return all_predictions, all_nbest_json


def postprocess_qa_predictions(
    examples,
    features,
    predictions: Tuple[np.ndarray, np.ndarray],
    n_best_size: int = 20,
    max_answer_length: int = 30,
    output_dir: Optional[str] = None,
    prefix: Optional[str] = None,
    is_world_process_zero: bool = True,
    do_pos_ensemble: bool = False,
):
    """
    Post-processes : qa model의 prediction 값을 후처리하는 함수
    모델은 start logit과 end logit을 반환하기 때문에, 이를 기반으로 original text로 변경하는 후처리가 필요함

    Args:
        examples: 전처리 되지 않은 데이터셋 (see the main script for more information).
        features: 전처리가 진행된 데이터셋 (see the main script for more information).
        predictions (:obj:`Tuple[np.ndarray, np.ndarray]`):
            모델의 예측값 :start logits과 the end logits을 나타내는 two arrays, 첫번째 차원은 :obj:`features`의 element와 갯수가 맞아야함.
        n_best_size (:obj:`int`, `optional`, defaults to 20):
            답변을 찾을 때 생성할 n-best prediction 총 개수
        max_answer_length (:obj:`int`, `optional`, defaults to 30):
            생성할 수 있는 답변의 최대 길이
        output_dir (:obj:`str`, `optional`):
            아래의 값이 저장되는 경로
            dictionary : predictions, n_best predictions (with their scores and logits) if:obj:`version_2_with_negative=True`,
            dictionary : the scores differences between best and null answers
        prefix (:obj:`str`, `optional`):
            dictionary에 `prefix`가 포함되어 저장됨
        is_world_process_zero (:obj:`bool`, `optional`, defaults to :obj:`True`):
            이 프로세스가 main process인지 여부(logging/save를 수행해야 하는지 여부를 결정하는 데 사용됨)
    """
    # Logging.
    logger.setLevel(logging.INFO if is_world_process_zero else logging.WARN)
    logger.info(
        f"Post-processing {len(examples)} example predictions split into {len(features)} features."
    )

    # get logits with checking predictions and features
    all_start_logits, all_end_logits = get_all_logits(predictions, features)

    # map features to example
    features_per_example = map_features_to_example(examples, features)

    # 전체 example들에 대한 main Loop
    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()

    for example_index, example in enumerate(tqdm(examples)):
        # 해당하는 현재 example index
        feature_indices = features_per_example[example_index]

        # 가장 좋은 `n_best_size` predictions만 유지합니다.
        predictions = get_candidate_preds(
            features, feature_indices, all_start_logits, all_end_logits, n_best_size, max_answer_length
        )

        # offset을 활용해 text로 변환 후, all_prediction, all_nbest_json 업데이트
        all_predictions, all_nbest_json = get_example_prediction(
            example, predictions, all_predictions, all_nbest_json, do_pos_ensemble
        )

    # output_dir이 있으면 모든 dicts를 저장합니다.
    if output_dir is not None:
        save_pred_json(
            all_predictions, all_nbest_json, output_dir, prefix
        )

    return all_predictions


# Post-processing:
def post_processing_function(
    examples,
    features,
    predictions,
    training_args,
    mode,
):
    if mode == 'predict' and training_args.do_pos_ensemble:
        training_args.do_pos_ensemble = True
    else:
        training_args.do_pos_ensemble = False

    # Post-processing: start logits과 end logits을 original context의 정답과 match시킵니다.
    predictions = postprocess_qa_predictions(
        examples=examples,
        features=features,
        predictions=predictions,
        max_answer_length=training_args.max_answer_length,
        output_dir=training_args.output_dir,
        prefix=training_args.run_name + '_' + mode,
        do_pos_ensemble=training_args.do_pos_ensemble
    )
    # Metric을 구할 수 있도록 Format을 맞춰줍니다.
    formatted_predictions = [
        {"id": k, "prediction_text": v} for k, v in predictions.items()
    ]
    if mode == "predict":
        return formatted_predictions
    else:
        references = [
            {"id": ex["id"], "answers": ex[ANSWER_COLUMN_NAME]}
            for ex in examples
        ]
        return EvalPrediction(
            predictions=formatted_predictions, label_ids=references
        )


def gen_postprocessing_function(examples, predictions, training_args, tokenizer):
    """
    postprocess는 nltk를 이용합니다.
    Huggingface의 TemplateProcessing을 사용하여
    정규표현식 기반으로 postprocess를 진행할 수 있지만
    해당 미션에서는 nltk를 이용하여 간단한 후처리를 진행합니다
    """
    import nltk
    nltk.download('punkt')

    if isinstance(predictions, tuple):
        predictions = predictions[0]

    # 후처리
    decoded_preds = tokenizer.batch_decode(
        predictions, skip_special_tokens=True)
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_preds = ["\n".join(nltk.sent_tokenize(pred))
                     for pred in decoded_preds]

    formatted_predictions = [
        {"id": ex["id"], "prediction_text": decoded_preds[i]} for i, ex in enumerate(examples)
    ]

    # 저장
    output_dir = training_args.output_dir
    assert os.path.isdir(output_dir), f"{output_dir} is not a directory."
    prediction_file = os.path.join(output_dir, "predictions.json")
    logger.info(f"Saving predictions to {prediction_file}.")
    with open(prediction_file, "w", encoding="utf-8") as writer:
        writer.write(
            json.dumps(formatted_predictions, indent=4,
                       ensure_ascii=False) + "\n"
        )

    # 결과 리턴
    if training_args.do_predict:
        return formatted_predictions

    elif training_args.do_eval:
        references = [
            {"id": ex["id"], "answers": ex[ANSWER_COLUMN_NAME]}
            for ex in examples
        ]
        return EvalPrediction(
            predictions=formatted_predictions, label_ids=references
        )
