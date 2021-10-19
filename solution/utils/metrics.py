from datasets import load_metric
from transformers import (
    EvalPrediction,
)
from utils.postprocessing import gen_postprocessing_function
import json

from solution.args import (
    HfArgumentParser,
    get_args_parser,
    DataArguments,
    ModelingArguments,
    NewTrainingArguments,
    ProjectArguments
)

def compute_metrics(p: EvalPrediction):
    metric = load_metric("squad")
    command_args = get_args_parser()
    parser = HfArgumentParser(
        (DataArguments, NewTrainingArguments, ModelingArguments, ProjectArguments)
    )
    data_args, _, model_args, _ = parser.parse_yaml_file(yaml_file=os.path.abspath(command_args.config))

    with open(data_args.dataset_name) as f:
        datasets = json.load(f)

    if model_args.method == 'gen':
        preds, labels = p
        if isinstance(preds, tuple):
            preds = preds[0]
        
        decoded_preds = model_args.tokenizer_name.batch_decode(preds, skip_special_tokens=True)
        # decoded_labels은 rouge metric을 위한 것이며, f1/em을 구할 때 사용되지 않음
        decoded_labels = model_args.tokenizer_name.batch_decode(labels, skip_special_tokens=True)

        # 간단한 post-processing
        decoded_preds, decoded_labels = gen_postprocessing_function(decoded_preds, decoded_labels)

        formatted_predictions = [{"id": ex["id"], "prediction_text": decoded_preds[i]} for i, ex in enumerate(datasets["validation"])]
        references = [{"id": ex["id"], "answers": ex["answers"]} for ex in datasets["validation"]]
        return metric.compute(predictions=formatted_predictions, references=references)
    return metric.compute(predictions=p.predictions, references=p.label_ids)