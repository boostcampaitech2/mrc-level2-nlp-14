from typing import List, Optional
from transformers import TrainingArguments
from dataclasses import dataclass, field


@dataclass
class NewTrainingArguments(TrainingArguments):
    """ Arguments related to training. """
    trainer_class: str = field(
        default="default",
        metadata={"help": "Trainer class name. You can find this object on `solution/trainers.__init__.py`."},
    ),
    loss : str = field(
        default="default",
        metadata={"help": "Which loss function to use for training."},
    )