from typing import Any

from transformers import AutoConfig, PreTrainedModel
from transformers import PreTrainedTokenizer

from .models import *
from ...args import ModelArguments

import sys

mod = sys.modules[__name__]


def _set_attr(main: Any, attrname: str, getfrom: Any):
    """ set value on configuration using other model_args """
    args_value = getattr(getfrom, attrname, None)
    args_value = getattr(main, attrname, args_value)
    setattr(main, attrname, args_value)


def _get_config(model_args: ModelArguments):
    """ Get configuration for hf models """
    config = AutoConfig.from_pretrained(
        model_args.config_name
        if model_args.config_name
        else model_args.model_name_or_path,
        use_auth_token=model_args.use_auth_token,
    )
    _set_attr(config, "reader_type", model_args)
    return config


def _get_model(
    model_args: ModelArguments,
    default_model: PreTrainedModel,
    config: AutoConfig,
):
    """ Get hf model. """
    model_cls = getattr(mod, model_args.architectures,
                        default_model)
    model = model_cls.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        use_auth_token=model_args.use_auth_token,
        revision=model_args.revision,
    )
    return model


def basic(
    model_args: ModelArguments,
    default_model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
):
    """ basic model initialize function.  """
    config = _get_config(model_args)
    model = _get_model(model_args, default_model, config)
    return model


def add_qaconv_head(
    model_args: ModelArguments,
    default_model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
):
    """ Init functions for Extractive Model + QA Conv head. """
    assert model_args.reader_type == "extractive"
    config = _get_config(model_args)
    if "sds_conv" in model_args.model_head:
        _set_attr(config, "qa_conv_input_size", model_args)
        _set_attr(config, "qa_conv_n_layers", model_args)
    elif "conv" in model_args.model_head:
        _set_attr(config, "qa_conv_out_channel", model_args)
        _set_attr(config, "model_head", model_args)
        setattr(config, "sep_token_id", tokenizer.sep_token_id)

    model = _get_model(model_args, default_model, config)
    return model


MODEL_INIT = {
    "basic": basic,
    "qaconv_head": add_qaconv_head,
}
