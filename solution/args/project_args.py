from typing import List, Optional
from dataclasses import dataclass, field


@dataclass
class ProjectArguments:
    """
    Arguments pertaining to construct project.
    """
    task: str = field(
        default="mrc",
        metadata={"help": "Task name. This kwarg is used by `TASK_INFOS_MAP` and `TASK_METRIC_MAP` to get task-specific information."},
    )
    wandb_project: str = field(
        default="mrc",
        metadata={"help": "weight and biases project name."},
    )
    save_model_dir: str = field(
        default="best",
        metadata={"help": "Directory where the trained model is stored."},
    )
    checkpoint: str = field(
        default=None,
        metadata={"help": "Checkpoint with models to be used for inference."},
    )