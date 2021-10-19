from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
)

from solution.reader.generative_models.modeling_bart import *
from solution.reader.generative_models.modeling_mt5 import *

from dataclasses import asdict

# Get __init__ modules
import sys

mod = sys.modules[__name__]


# Get model
def gen_model_init(model_args, task_infos, tokenizer):
    """ Initialization function for basic models """
    config = AutoConfig.from_pretrained(
                model_args.config_name
                if model_args.config_name
                else model_args.model_name_or_path,
            )
    for key, value in asdict(model_args).items():
            setattr(config, key, value)
    model_cls = getattr(mod, model_args.architectures,
                        AutoModelForSeq2SeqLM)
    model = model_cls.from_pretrained(
        pretrained_model_name_or_path=model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
    )
    return model


GEN_MODEL_INIT_FUNC = {
    "basic": gen_model_init,
}