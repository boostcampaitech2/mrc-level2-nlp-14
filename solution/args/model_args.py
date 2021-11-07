from typing import List, Optional
from dataclasses import dataclass, field

from .base import ModelArguments


@dataclass
class ModelingArguments(ModelArguments):
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: str = field(
        default="klue/bert-base",
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    reader_type: str = field(
        default="extractive",
        metadata={
            "help": "Method to MRC system based in.",
            "choices": ["extractive", "generative", "ensemble"],
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
    use_auth_token: bool = field(
        default=False,
        metadata={"help": "Decide whether to use the auth token"}
    )
    revision: str = field(
        default="main",
        metadata={"help": "Determine the version of the model"}
    )


@dataclass
class ModelHeadArguments(ModelingArguments):
    model_head: str = field(
        default="conv",
        metadata={
            "help": "Decide which head to use in the MRC model",
            "choices": ["conv"],
        },
    )
    qa_conv_out_channel: int = field(
        default=1024,
        metadata={"help": "ConvLayer out channel"},
    )
    qa_conv_input_size: int = field(
        default=512,
        metadata={
            "help": "Determine the sequence length to receive input from QA SDS Head. "
            "The input sequence length of the model must be padded unconditionally."
        },
    )
    qa_conv_n_layers: int = field(
        default=5,
        metadata={"help": "Decide how deep to stack the layers in QA SDS Head."},
    )


@dataclass
class MrcModelArguments(ModelHeadArguments):
    pass
