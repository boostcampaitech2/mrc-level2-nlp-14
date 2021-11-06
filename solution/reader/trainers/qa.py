import time
import math
from typing import Optional, List, Dict, Callable

from transformers import (
    is_datasets_available,
    is_torch_tpu_available,
)
from transformers.trainer_utils import (
    PredictionOutput,
    speed_metrics,
    denumpify_detensorize,
)
from transformers.debug_utils import DebugOption

from .base import BaseTrainer


if is_datasets_available():
    import datasets

if is_torch_tpu_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met


class QuestionAnsweringTrainer(BaseTrainer):

    def __init__(
        self,
        *args,
        eval_examples: datasets.Dataset = None,
        post_process_function: Callable = None,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.eval_examples = eval_examples
        self.post_process_function = post_process_function

    def evaluate(
        self,
        eval_dataset: Optional[datasets.Dataset] = None,
        eval_examples: Optional[datasets.Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
        mode: str = "evaluate",
    ) -> Dict[str, float]:
        # memory metrics - must set up as early as possible
        self._memory_tracker.start()
        eval_dataset = self.eval_dataset if eval_dataset is None else eval_dataset
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        start_time = time.time()
        eval_examples = self.eval_examples if eval_examples is None else eval_examples
        # 일시적으로 metric computation를 불가능하게 한 상태이며, 해당 코드에서는 loop 내에서 metric 계산을 수행합니다.
        compute_metrics = self.compute_metrics
        self.compute_metrics = None
        eval_loop = self.prediction_loop if self.args.use_legacy_prediction_loop else self.evaluation_loop
        try:
            output = eval_loop(
                eval_dataloader,
                description="Evaluation",
                # metric이 없으면 예측값을 모으는 이유가 없으므로 아래의 코드를 따르게 됩니다.
                prediction_loss_only=True if compute_metrics is None else None,
                ignore_keys=ignore_keys,
                metric_key_prefix=metric_key_prefix,
            )
        finally:
            self.compute_metrics = compute_metrics

        if isinstance(eval_dataset, datasets.Dataset):
            eval_dataset.set_format(
                type=eval_dataset.format["type"],
                columns=list(eval_dataset.features.keys()),
            )

        if self.post_process_function is not None and self.compute_metrics is not None:
            eval_preds = self.post_process_function(
                eval_examples,
                eval_dataset,
                output.predictions,
                self.args,
                mode,
            )
            metrics = self.compute_metrics(eval_preds)

            # To be JSON-serializable, we need to remove numpy types or zero-d tensors
            metrics = denumpify_detensorize(metrics)

            # Prefix all keys with metric_key_prefix + '_'
            for key in list(metrics.keys()):
                if not key.startswith(f"{metric_key_prefix}_"):
                    metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

            total_batch_size = self.args.eval_batch_size * self.args.world_size
            metrics.update(
                speed_metrics(
                    metric_key_prefix,
                    start_time,
                    num_samples=output.num_samples,
                    num_steps=math.ceil(output.num_samples / total_batch_size),
                )
            )
            self.log(metrics)
        else:
            metrics = {}

        if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
            # tpu-comment: PyTorch/XLA에 대한 Logging debug metrics (compile, execute times, ops, etc.)
            xm.master_print(met.metrics_report())

        self.control = self.callback_handler.on_evaluate(
            self.args, self.state, self.control, metrics
        )

        self._memory_tracker.stop_and_update_metrics(metrics)

        return metrics

    def predict(
        self,
        test_dataset: datasets.Dataset,
        test_examples: datasets.Dataset,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "test",
        mode: str = "predict",
    ) -> PredictionOutput:
        if mode.lower in ["test", "pred", "predict"]:
            mode = "predict"
        assert mode == "predict"
        # memory metrics - must set up as early as possible
        # self._memory_tracker.start()

        test_dataloader = self.get_test_dataloader(test_dataset)
        start_time = time.time()

        # 일시적으로 metric computation를 불가능하게 한 상태이며, 해당 코드에서는 loop 내에서 metric 계산을 수행합니다.
        # evaluate 함수와 동일하게 구성되어있습니다
        compute_metrics = self.compute_metrics
        self.compute_metrics = None
        eval_loop = self.prediction_loop if self.args.use_legacy_prediction_loop else self.evaluation_loop
        try:
            output = eval_loop(
                test_dataloader,
                description="Prediction",
                ignore_keys=ignore_keys,
                metric_key_prefix=metric_key_prefix,
            )
        finally:
            self.compute_metrics = compute_metrics

        if self.post_process_function is None or self.compute_metrics is None:
            return output

        if isinstance(test_dataset, datasets.Dataset):
            test_dataset.set_format(
                type=test_dataset.format["type"],
                columns=list(test_dataset.features.keys()),
            )

        predictions = self.post_process_function(
            test_examples,
            test_dataset,
            output.predictions,
            self.args,
            mode,
        )

        # self._memory_tracker.stop_and_update_metrics(output.metrics)

        return predictions
