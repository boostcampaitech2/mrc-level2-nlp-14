from transformers import (
    AutoConfig,
    AutoModelForQuestionAnswering,
)

from solution.reader.extractive_models.modeling_bart import *
from solution.reader.extractive_models.modeling_bert import *

from dataclasses import asdict

# Get __init__ modules
import sys

mod = sys.modules[__name__]


# Get model
def ext_model_init(model_args):
    """ Initialization function for basic models """
    config = AutoConfig.from_pretrained(
                model_args.config_name
                if model_args.config_name
                else model_args.model_name_or_path,
            )
    for key, value in asdict(model_args).items():
            setattr(config, key, value)
    model_cls = getattr(mod, model_args.architectures,
                        AutoModelForQuestionAnswering)
    model = model_cls.from_pretrained(
        pretrained_model_name_or_path=model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
    )
    return model


EXT_MODEL_INIT_FUNC = {
    "basic": ext_model_init,
}