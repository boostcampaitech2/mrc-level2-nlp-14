from .odqa import OdqaProcessor, convert_examples_to_features
from .post import post_processing_function, gen_postprocessing_function


POST_PROCESSING_FUNCTION = {
    "extractive": post_processing_function,
    "generative": gen_postprocessing_function,
    "ensemble": gen_postprocessing_function,
}
