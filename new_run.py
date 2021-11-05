import os
import sys
import json
import torch

from solution.args import (
    HfArgumentParser,
    MrcDataArguments,
    MrcModelArguments,
    MrcTrainingArguments,
    MrcProjectArguments,
)
from solution.data.metrics import compute_metrics
from solution.data.processors import (
    OdqaProcessor,
    convert_examples_to_features,
    post_processing_function,
)
from solution.reader import READER_HOST
from solution.retrieval import RETRIEVAL_HOST
from solution.utils import set_seed, check_no_error

from transformers import AutoTokenizer
from transformers.utils import logging

from solution.data.processors.mask import make_emb_dataset

logging.set_verbosity_info()
logger = logging.get_logger(__name__)

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["TRANSFORMERS_VERBOSITY"] = "info"


def main():
    parser = HfArgumentParser(
            [
            MrcDataArguments,
            MrcModelArguments,
            MrcTrainingArguments,
            MrcProjectArguments
            ]
    )
    if len(sys.argv) == 2 and sys.argv[1].endswith(".yaml"):
        args = parser.parse_yaml_file(yaml_file=os.path.abspath(sys.argv[1]))
    else:
        args = parser.parse_args_into_dataclasses()
    data_args, model_args, training_args, project_args = args

    set_seed(training_args.seed)

    if data_args.make_mask:
        make_emb_dataset(data_args.dataset_path,data_args.masking_type)
        return
    print(f"model is from {model_args.model_name_or_path}")
    print(f"data is from {data_args.dataset_path}")

    # wandb setting
    os.environ["WANDB_PROJECT"] = project_args.wandb_project

    # Load Processor
    processor = OdqaProcessor(data_args, model_args, training_args)

    # Load Retriever
    retriever_cls = RETRIEVAL_HOST[data_args.retrieval_mode][data_args.retrieval_name]
    retriever = retriever_cls(data_args)

    # Load Reader
    reader_cls = READER_HOST[model_args.reader_type]
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name is not None
        else model_args.model_name_or_path,
        use_auth_token=model_args.use_auth_token,
        revision=model_args.revision,
    )
    reader = reader_cls(model_args, tokenizer)

    train_features, train_datasets = convert_examples_to_features(processor, tokenizer)

    eval_features, eval_datasets = convert_examples_to_features(
        processor, tokenizer, mode="eval")
    
    reader.set_trainer(
        model_init=reader.model_init,
        args=training_args,
        train_dataset=train_features,
        eval_dataset=eval_features,
        eval_examples=eval_datasets,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        post_process_function=post_processing_function,
    )

    last_checkpoint, data_args.max_seq_length = check_no_error(
        data_args, training_args, tokenizer,
    )
    logger.warning(f"LAST CHECKPOINT: {last_checkpoint}")

    # checkpoint setting
    checkpoint = project_args.checkpoint
    if checkpoint is None:
        if last_checkpoint is not None:
            checkpoint = last_checkpoint
        elif os.path.isdir(reader.model_args.model_name_or_path):
            checkpoint = model_args.model_name_or_path
    logger.warning(f"CHECKPOINT: {checkpoint}")


    #curriculum learning setting
    if data_args.curriculum_learn:
        logger.warning(f"load from checkpoint {checkpoint}")
        ckpt_model_file = os.path.join(checkpoint, "pytorch_model.bin")
        state_dict = torch.load(ckpt_model_file, map_location="cpu")
        reader._trainer._load_state_dict_in_model(state_dict)
        del state_dict
        torch.cuda.empty_cache()
        checkpoint = None

    if training_args.do_train:
        with reader.mode_change(mode="train"):
            train_results = reader.read(resume_from_checkpoint=checkpoint)
            reader.save_trainer()
            reader.save_metrics("train", 
                                train_results.metrics, 
                                train_datasets)
            checkpoint = training_args.output_dir

    if training_args.do_eval:
        if data_args.eval_retrieval:
            eval_features, eval_datasets = convert_examples_to_features(
                processor, tokenizer, retriever, topk=data_args.top_k_retrieval, mode="eval")
    
        logger.warning(f"load from checkpoint {checkpoint}")
        ckpt_model_file = os.path.join(checkpoint, "pytorch_model.bin")
        state_dict = torch.load(ckpt_model_file, map_location="cpu")
        reader._trainer._load_state_dict_in_model(state_dict)
        del state_dict
        torch.cuda.empty_cache()
        
        with reader.mode_change(mode="evaluate"):
            eval_metrics = reader.read(eval_dataset=eval_features,
                                    eval_examples=eval_datasets,
                                    mode=reader.mode)
            reader.save_metrics("eval",
                                eval_metrics,
                                eval_datasets)

    if training_args.do_predict & data_args.eval_retrieval:
        test_features, test_datasets = convert_examples_to_features(
            processor, tokenizer, retriever, topk=data_args.top_k_retrieval, mode="test")
    
        with reader.mode_change(mode="predict"):
            pred_results = reader.read(test_dataset=test_features,
                                    test_examples=test_datasets,
                                    mode=reader.mode)

    # 오답노트 기능을 사용할 경우 Analyzer로 분석
    # if project_args.report_to_wrong_answers:
    #     report = Analyzer.make_report(eval_result)
    #     Analyzer.post(report)


if __name__ == "__main__":   
    main()
