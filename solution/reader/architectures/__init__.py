from transformers import AutoConfig, PreTrainedModel
from transformers import PreTrainedTokenizer

from .models import *
from ...args import ModelArguments

import sys

mod = sys.modules[__name__]


def _get_config(model_args: ModelArguments):
    config = AutoConfig.from_pretrained(
        model_args.config_name
        if model_args.config_name
        else model_args.model_name_or_path,
    )
    return config


def _get_model(
    model_args: ModelArguments,
    default_model: PreTrainedModel,
    config: AutoConfig,
):
    model_cls = getattr(mod, model_args.architectures,
                        default_model)
    model = model_cls.from_pretrained(
        model_args.model_name_or_path,
        config=config,
    )
    return model


def basic(
    model_args: ModelArguments, 
    default_model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
):
    config = _get_config(model_args)
    model = _get_model(model_args, default_model, config)
    return model


def add_qaconv_head(
    model_args: ModelArguments,
    default_model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
):
    assert model_args.reader_type != "extractive"
    config = _get_config(model_args)
    config.model_head = model_args.model_head
    if "conv" in config.model_head == :
        config.qa_conv_out_channel = model_args.qa_conv_out_channel
        config.qa_conv_input_size = model_args.qa_conv_input_size
        config.qa_conv_n_layers = model_args.qa_conv_n_layers
    config.sep_token_id = tokenizer.sep_token_id
    model = _get_model(model_args, default_model, config)
    return model


MODEL_INIT = {
    "basic": basic,
    "qaconv_head": add_qaconv_head,
}