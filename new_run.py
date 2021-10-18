import logging
import os
import sys
from functools import partial
from typing import List, Callable

import wandb
from datasets import Sequence, Value, Features, Dataset, DatasetDict


from transformers import set_seed

from retrieval import SparseRetrieval

from solution.args import (
    HfArgumentParser,
    get_args_parser,
    DataArguments,
    TrainingArguments,
    NewTrainingArguments,
    ModelingArguments,
    ProjectArguments
)
from solution.reader import (
    post_processing_function,
    ExtractiveReader,
    GenerativeReader,
    ext_prepare_features,
    gen_prepare_features
)
from solution.utils import (
    compute_metrics,
)


def main():
    # 가능한 arguments 들은 ./arguments.py 나 transformer package 안의 src/transformers/training_args.py 에서 확인 가능합니다.
    # --help flag 를 실행시켜서 확인할 수 도 있습니다.
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -    %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        level=logging.INFO,
    )



    command_args = get_args_parser()
    parser = HfArgumentParser(
        (DataArguments, NewTrainingArguments, ModelingArguments, ProjectArguments)
    )
    data_args, training_args, model_args, project_args = \
        parser.parse_yaml_file(yaml_file=os.path.abspath(command_args.config))

    # Set-up WANDB
    os.environ["WANDB_PROJECT"] = project_args.wandb_project
    # Reader 모델 통합 관리 객체. 생성시에 데이터셋 및 모델 세팅 수행됨
    if model_args.method == 'ext':
        reader = ExtractiveReader(command_args=command_args,
                                    compute_metrics=compute_metrics,
                                    pre_process_function=ext_prepare_features,
                                    post_process_function=post_processing_function,
                                    logger=logger)
    elif model_args.method == 'gen':
        reader = GenerativeReader(command_args=command_args,
                                    compute_metrics=compute_metrics,
                                    pre_process_function=gen_prepare_features,
                                    post_process_function=post_processing_function,
                                    logger=logger)
    else:
        raise ValueError("Check whether model_args.method is 'ext or 'gen'")

    '''
    1. answer 존재 유무 : validation of train set / validation of test_set
    2. context retrieval 유무 : eval_retrieval True / False
    
    - reader 모델의 성능 평가                : train_dataset['valid'], eval_retrieval = False
    - reader와 retrieval 모델 조합의 성능 평가 : train_dataset['valid'], eval_retrieval = True
    - submission 파일 생성                  : test_dataset['valid'], eval_retrieval = True
    '''

    # Trainer 객체 설정. Retireved Dataset이 Predict를 위해 주어졌을 때, 기존 저장된 eval_dataset과 swap
    reader.set_trainer()
    print(reader.model)
    print(f"model is from {reader.args.model_args.model_name_or_path}")
    print(f"data is from {reader.args.data_args.dataset_name}")

    # do_train mrc model 혹은 do_eval mrc model
    if reader.args.training_args.do_train:
        if reader.last_checkpoint is not None:
            checkpoint = reader.last_checkpoint
        elif os.path.isdir(reader.args.model_args.model_name_or_path):
            checkpoint = reader.args.model_args.model_name_or_path
        else:
            checkpoint = None

        train_result = reader.trainer.train(resume_from_checkpoint=checkpoint)
        reader.trainer.save_model() # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        metrics["train_samples"] = len(reader.trainer.train_dataset)

        reader.trainer.log_metrics("train", metrics)
        reader.trainer.save_metrics("train", metrics)
        reader.trainer.save_state()

        output_train_file = os.path.join(reader.args.training_args.output_dir, "train_results.txt")
        
        with open(output_train_file, "w") as writer:
            logger.info("***** Train results *****")
            for key, value in sorted(train_result.metrics.items()):
                logger.info(f"  {key} = {value}")
                writer.write(f"{key} = {value}\n")

        # State 저장
        reader.trainer.state.save_to_json(
            os.path.join(reader.args.training_args.output_dir, "trainer_state.json")
        )


        
    # Evaluation
    # train_dataset['validation']으로 성능을 평가합니다. retrieval 활용 여부에 따라
    # 1. eval_retrieval == False : MRC Model의 단독 성능 평가
    # 2. eval_retrieval == True  : MRC Model & Retrieval Model 조합의 성능 평가
    # See solution.reader.postprocessing L326-336
    if reader.args.training_args.do_eval:
        # do_eval, do_predict 둘 모두 True면 do_eval이 되지 않아 do_predict를 임시로 끔
        do_predict = reader.args.training_args.do_predict
        if reader.args.training_args.do_predict:
            reader.args.training_args.do_predict = False

        retrieved_dataset = None
        retrieved_examples = None
        if reader.args.data_args.eval_retrieval:
            logger.info("*** Evaluate with Retrieved passage ***")
            retrieved_examples = run_sparse_retrieval(
                tokenize_fn=reader.tokenizer.tokenize,
                datasets=reader.datasets,
                training_args=reader.args.training_args,
                data_args=reader.args.data_args,
                )
            retrieved_examples = retrieved_examples["validation"]
            retrieved_dataset = reader.preprocessing_retrieved_doc(retrieved_examples)

        logger.info("*** Evaluate ***")
        metrics = reader.trainer.evaluate(eval_dataset=retrieved_dataset,
                                          eval_examples=retrieved_examples)
        metrics["eval_samples"] = len(reader.eval_dataset)
        reader.trainer.log_metrics("eval", metrics)
        reader.trainer.save_metrics("eval", metrics)

        if do_predict:
            reader.args.training_args.do_predict = True
    
    # Submission
    if reader.args.training_args.do_predict:
        logger.info("*** Predict ***")
        if not reader.args.data_args.eval_retrieval:
            raise ValueError('*** For submission, you must use retireval model(set --eval_retrieval True --do_predict True ***')


        retrieved_dataset = None
        retrieved_examples = None
        retrieved_examples = run_sparse_retrieval(
            tokenize_fn=reader.tokenizer.tokenize,
            datasets=reader.test_datasets,
            training_args=reader.args.training_args,
            data_args=reader.args.data_args,
            )
        retrieved_examples = retrieved_examples["validation"]
        retrieved_dataset = reader.preprocessing_retrieved_doc(retrieved_examples)

        assert retrieved_dataset is not None
        assert retrieved_examples is not None

        predictions = reader.trainer.predict(
            test_dataset=retrieved_dataset, test_examples=retrieved_examples
        )
        # predictions.json 은 postprocess_qa_predictions() 호출시 이미 저장됩니다.
        print(
            "No metric can be presented because there is no correct answer given. Job done!"
        )

# ====================================================
# TODO 바꿔줄 함수
def run_sparse_retrieval(
    tokenize_fn: Callable[[str], List[str]],
    datasets: DatasetDict,
    training_args: NewTrainingArguments,
    data_args: DataArguments,
    data_path: str = "../data",
    context_path: str = "wikipedia_documents.json",
) -> DatasetDict:

    # Query에 맞는 Passage들을 Retrieval 합니다.
    retriever = SparseRetrieval(
        tokenize_fn=tokenize_fn, data_path=data_path, context_path=context_path
    )
    retriever.get_sparse_embedding()

    if data_args.use_faiss:
        retriever.build_faiss(num_clusters=data_args.num_clusters)
        df = retriever.retrieve_faiss(
            datasets["validation"], topk=data_args.top_k_retrieval
        )
    else:
        df = retriever.retrieve(datasets["validation"], topk=data_args.top_k_retrieval)

    # test data 에 대해선 정답이 없으므로 id question context 로만 데이터셋이 구성됩니다.
    if training_args.do_predict:
        f = Features(
            {
                "context": Value(dtype="string", id=None),
                "id": Value(dtype="string", id=None),
                "question": Value(dtype="string", id=None),
            }
        )

    # train data 에 대해선 정답이 존재하므로 id question context answer 로 데이터셋이 구성됩니다.
    elif training_args.do_eval:
        f = Features(
            {
                "answers": Sequence(
                    feature={
                        "text": Value(dtype="string", id=None),
                        "answer_start": Value(dtype="int32", id=None),
                    },
                    length=-1,
                    id=None,
                ),
                "context": Value(dtype="string", id=None),
                "id": Value(dtype="string", id=None),
                "question": Value(dtype="string", id=None),
            }
        )
    datasets = DatasetDict({"validation": Dataset.from_pandas(df, features=f)})
    return datasets


if __name__ == "__main__":
    main()
