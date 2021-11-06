from typing import List, Optional
from dataclasses import dataclass, field

from .base import ProjectArguments


@dataclass
class AnalyzerArguments(ProjectArguments):
    wandb_project: str = field(
        default="mrc",
        metadata={"help": "weight and biases project name."},
    )


@dataclass
class MrcProjectArguments(AnalyzerArguments):
    checkpoint: str = field(
        default=None,
        metadata={"help": "checkpoint directory path"}
    )
