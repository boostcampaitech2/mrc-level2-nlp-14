from typing import List, Optional
from dataclasses import dataclass, field

from .base import ProjectArguments


"""
PROJECT ARGS에 ANALYZER에 대한 세팅을 추가해야 한다.
"""

@dataclass
class AnalyzerArguments(ProjectArguments):
    wandb_project: str = field(
        default="mrc",
        metadata={"help": "weight and biases project name."},
    )
    
    
@dataclass
class MrcProjectArguments(AnalyzerArguments):
    pass