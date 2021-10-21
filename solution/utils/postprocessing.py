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
"""
Pre-processing
Post-processing utilities for question answering.
"""
import collections
import json
import logging
import os
from typing import Optional, Tuple

import numpy as np
from tqdm.auto import tqdm

from transformers import EvalPrediction
from solution.args import (
    HfArgumentParser,
    get_args_parser,
    DataArguments,
    ModelingArguments,
    NewTrainingArguments,
    ProjectArguments,
)
from solution.utils.constant import (
    ANSWER_COLUMN_NAME,
)

logger = logging.getLogger(__name__)

def save_pred_json(
    all_predictions, all_nbest_json, output_dir, prefix
):
    """
    output_dir에 prediction.json, nbest_predctions.json을 저장합니다.
    
    Args:
        all_predictions ([type]): [description]
        all_nbest_json ([type]): [description]
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
    predictions과 features length에 대해 assertions을 체크한 후,
    start & end logtis([ndarray], [ndarray])을 리턴합니다.
    
    Args:
        predictions ([Tuple[ndarray, ndarray]]): start & end logit predictions
        features ([Dataset]): tokenized & splited datasets
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
    exmaple index에 feature indices를 맵핑하여,
    Dict(key : exmaples index, value : feature indices) 값으로 리턴합니다.
    
    Args:
        examples ([Dataset]): raw datasets
        features ([Dataset]): tokenized & splited datasets
    """
    
    # example과 mapping되는 feature 생성
    example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
    features_per_example = collections.defaultdict(list)
    for i, feature in enumerate(features):
        features_per_example[example_id_to_index[feature["example_id"]]].append(i)
    
    return features_per_example


def get_candidate_preds(
    features, feature_indices, all_start_logits, all_end_logits, n_best_size, max_answer_length
):
    """
    한 exmaple에 맵핑된 features 중 n_best_size만큼의
    prediction([List[Dict(key : (offset, score, start_logit, end_logit)])을 리턴합니다.

    Args:
        features ([Dataset]): tokenized & splited datasets
        feature_indices ([List]): feature indices of one loop exmaple
        all_start_logits ([ndarray]): all start logits
        all_end_logits ([ndarray]): all end logits
        n_best_size ([int]): number of return best predictions
        max_answer_length ([int]): max span of answer
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
            -1 : -n_best_size - 1 : -1
        ].tolist()

        end_indexes = np.argsort(end_logits)[
            -1 : -n_best_size - 1 : -1
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
    example, predictions, all_predictions, all_nbest_json
):  
    """
    한 exmaple에서 나온 prediction으로부터 offset을 answer text로 변환 후,
    all_predictions[List[Dict(key : (score, start_logit, end_logit, text)],
    all_nbest_json에 prediction을 추가하여 리턴합니다.
    
    Args:
        example ([Dataset]): raw datasets
        predictions ([List[Dict]]): prediction of one example
        all_predictions ([Dict]): total prediction to be updated
        all_nbest_json ([Dict]): total prediction of nbest size to be updated
    """
    # predict text offset mapping
    context = example["context"]
    for pred in predictions:
        offsets = pred.pop("offsets")
        pred["text"] = context[offsets[0] : offsets[1]]

    # rare edge case에는 null이 아닌 예측이 하나도 없으며 failure를 피하기 위해 fake prediction을 만듭니다.
    if len(predictions) == 0 or (
        len(predictions) == 1 and predictions[0]["text"] == ""
    ):

        predictions.insert(
            0, {"text": "empty", "start_logit": 0.0, "end_logit": 0.0, "score": 0.0}
        )

    # 모든 점수의 소프트맥스를 계산합니다(we do it with numpy to stay independent from torch/tf in this file, using the LogSumExp trick).
    scores = np.array([pred.pop("score") for pred in predictions])
    exp_scores = np.exp(scores - np.max(scores))
    probs = exp_scores / exp_scores.sum()

    # 예측값에 확률을 포함합니다.
    for prob, pred in zip(probs, predictions):
        pred["probability"] = prob

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
    is_world_process_zero: bool = True, ##
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
            example, predictions, all_predictions, all_nbest_json
        )
        
    # output_dir이 있으면 모든 dicts를 저장합니다.
    if output_dir is not None:
        save_pred_json(
            all_predictions, all_nbest_json, output_dir, prefix
        )

    return all_predictions


# Post-processing:
def post_processing_function(examples, features, predictions, training_args):
    command_args = get_args_parser()
    parser = HfArgumentParser(
        (DataArguments, NewTrainingArguments, ModelingArguments, ProjectArguments)
    )
    data_args, _, _, _ = \
        parser.parse_yaml_file(yaml_file=os.path.abspath(command_args.config))

    # Post-processing: start logits과 end logits을 original context의 정답과 match시킵니다.
    predictions = postprocess_qa_predictions(
        examples=examples,
        features=features,
        predictions=predictions,
        max_answer_length=data_args.max_answer_length,
        output_dir=training_args.output_dir,
    )
    # Metric을 구할 수 있도록 Format을 맞춰줍니다.
    formatted_predictions = [
        {"id": k, "prediction_text": v} for k, v in predictions.items()
    ]
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