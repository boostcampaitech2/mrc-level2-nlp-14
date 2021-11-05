from typing import List, Optional
from dataclasses import dataclass, field

from .base import ModelArguments


"""
method -> reader_type
    ext -> extractive
    gen -> generative
    
head -> model_head
conv_out_channel -> qa_conv_out_channel
"""

@dataclass
class ModelingArguments(ModelArguments):
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: str = field(
        default="klue/bert-base",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    reader_type: str = field(
        default="extractive",
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
    use_auth_token: bool = field(
        default=False,
        metadata={"help": ""}
    )
    revision: str = field(
        default="main",
        metadata={"help": ""}
    )


@dataclass
class HighlightingArguments(ModelArguments):

    underline: bool = field(
        default=False,
        metadata={"help": "Whether to add underline embedding at the time of tokenizing or not"},
    )
    punctuation: bool = field(
        default=False,
        metadata={"help": "Whether to add punctuation or not"},
    )
    punct_model_name_or_path: str = field(
        default="'kiyoung2/roberta-large-qaconv-sds-aug'",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    max_seq_length: int = field(
        default=100,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    punct_use_auth_token: bool = field(
        default=False,
        metadata={"help": ""}
    )
    punct_revision: str = field(
        default="main",
        metadata={"help": ""}
    )
    top_k_retrieval: int = field(
        default=10,
        metadata={
            "help": "Define how many top-k sentences to retrieve based on similarity."
        },
    )

    
@dataclass
class ModelHeadArguments(ModelingArguments):
    # 어떻게 구분지을지 고민하기
    model_head: str = field(
        default="conv",
        metadata={"help": "Extractive Model Head"},
    )
    qa_conv_out_channel: int = field(
        default=1024,
        metadata={"help": "ConvLayer out channel"},
    )
    qa_conv_input_size: int = field(
        default=512,
        metadata={"help":""},
    )
    qa_conv_n_layers: int = field(
        default=5,
        metadata={"help":""},
    )
    
    
@dataclass
class MrcModelArguments(ModelHeadArguments):
    pass