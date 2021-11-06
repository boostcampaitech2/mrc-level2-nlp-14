from typing import Optional, List, Dict, Callable, Union, Any, Tuple

import torch
import torch.nn as nn
from packaging import version

from transformers import (
    is_datasets_available,
)
from .seq2seq_qa import QuestionAnsweringSeq2SeqTrainer


if is_datasets_available():
    import datasets

if version.parse(torch.__version__) >= version.parse("1.6"):
    from torch.cuda.amp import autocast

_hf_deepspeed_config_weak_ref = None


def is_deepspeed_zero3_enabled():
    if _hf_deepspeed_config_weak_ref is not None and _hf_deepspeed_config_weak_ref() is not None:
        return _hf_deepspeed_config_weak_ref().is_zero3()
    else:
        return False


class QuestionAnsweringEnsembleTrainer(QuestionAnsweringSeq2SeqTrainer):

    def __init__(
        self,
        *args,
        eval_examples: datasets.Dataset = None,
        post_process_function: Callable = None,
        **kwargs
    ):
        super().__init__(*args, eval_examples=eval_examples,
                         post_process_function=post_process_function, **kwargs)
        self.label_names = ["start_positions", "end_positions", "labels"]

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:

        if not self.args.predict_with_generate or prediction_loss_only:
            return super().prediction_step(
                model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
            )

        has_labels = "labels" in inputs
        inputs = self._prepare_inputs(inputs)

        # XXX: adapt synced_gpus for fairscale as well
        gen_kwargs = {
            "max_length": self._max_length if self._max_length is not None else self.model.config.max_length,
            "num_beams": self._num_beams if self._num_beams is not None else self.model.config.num_beams,
            "synced_gpus": True if is_deepspeed_zero3_enabled() else False,
        }

        generated_tokens = self.model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            **gen_kwargs,
        )
        # in case the batch is shorter than max length, the output should be padded
        if generated_tokens.shape[-1] < gen_kwargs["max_length"]:
            generated_tokens = self._pad_tensors_to_max_len(
                generated_tokens, gen_kwargs["max_length"])

        with torch.no_grad():
            if self.use_amp:
                with autocast():
                    outputs = model(**inputs)
            else:
                outputs = model(**inputs)
            if has_labels:
                if self.label_smoother is not None:
                    loss = self.label_smoother(
                        outputs, inputs["labels"]).mean().detach()
                elif "loss" in outputs:
                    loss = (outputs["loss"] if isinstance(
                        outputs, dict) else outputs[0]).mean().detach()
                else:
                    loss = None
            else:
                loss = None

        if self.args.prediction_loss_only:
            return (loss, None, None)

        if has_labels:
            labels = inputs["labels"]
            if labels.shape[-1] < gen_kwargs["max_length"]:
                labels = self._pad_tensors_to_max_len(
                    labels, gen_kwargs["max_length"])
        else:
            labels = None

        return (loss, generated_tokens, labels)
