from datasets import load_metric
from transformers import EvalPrediction


def compute_metrics(p: EvalPrediction):
    metric = load_metric("squad")
    return metric.compute(predictions=p.predictions, references=p.label_ids)