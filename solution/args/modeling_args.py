from typing import List, Optional
from dataclasses import dataclass, field


@dataclass
class ModelingArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: str = field(
        default="klue/bert-base",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    method: str = field(
        default="ext",
        metadata={
            "help": "Method to MRC system based in. e.g. ext : extraction-based, gen : generation-based"
        },
    )
    architectures: str = field(
        default="ExtractiveReaderBaselineModel",
        metadata={"help": "Reader Model Archtectures"},
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )
    model_cache_dir: str = field(
        default="cache",
        metadata={"help": "Model cache directory path"},
    )
    model_init: str = field(
        default="basic",
        metadata={"help": "Which function to use to initialize the model?"},
    )