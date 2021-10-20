from typing import List, Optional
from transformers import TrainingArguments
from dataclasses import dataclass, field


@dataclass
class NewTrainingArguments(TrainingArguments):
    """ Arguments related to training. 
    
    """
    # trainer_class: str = field(
    #     default="default",
    #     metadata={"help": "Trainer class name. You can find this object on `solution/trainers.__init__.py`."},
    # ),
    # loss : str = field(
    #     default="default",
    #     metadata={"help": "Which loss function to use for training."},
    # )
    sortish_sampler: bool = field(default=False, metadata={"help": "Whether to use SortishSampler or not."})
    predict_with_generate: bool = field(
        default=False, metadata={"help": "Whether to use generate to calculate generative metrics (ROUGE, BLEU)."}
    )
    generation_max_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "The `max_length` to use on each evaluation loop when `predict_with_generate=True`. Will default "
            "to the `max_length` value of the model configuration."
        },
    )
    generation_num_beams: Optional[int] = field(
        default=None,
        metadata={
            "help": "The `num_beams` to use on each evaluation loop when `predict_with_generate=True`. Will default "
            "to the `num_beams` value of the model configuration."
        },
    )