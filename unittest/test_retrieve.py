import os
import sys

from transformers import AutoTokenizer
from datasets import load_from_disk, concatenate_datasets

# Add project path
abs_path = os.path.abspath(__file__) # ./test_retrieve.py
src_path = os.path.dirname(abs_path)
project_path = os.path.dirname(src_path)
sys.path.append(project_path)

from solution.args import HfArgumentParser, DataArguments, ModelingArguments
from solution.retrieval import TfidfRetrieval


def main(args):
    # Test sparse
    org_dataset = load_from_disk(args.dataset_name)
    full_ds = concatenate_datasets(
        [
            org_dataset["train"].flatten_indices(),
            org_dataset["validation"].flatten_indices(),
        ]
    )  # train dev 를 합친 4192 개 질문에 대해 모두 테스트
    print("*" * 40, "query dataset", "*" * 40)
    print(full_ds)
    
    retriever = TfidfRetrieval(args)
    
    query = "대통령을 포함한 미국의 행정부 견제권을 갖는 국가 기관은?"
    
    if args.use_faiss:

        # test single query
        scores, indices = retriever.retrieve_faiss(query)

        # test bulk
        df = retriever.retrieve_faiss(full_ds, topk=args.top_k_retrieval)
        df["correct"] = df["original_context"] == df["context"]

        print("correct retrieval result by faiss", df["correct"].sum() / len(df))

    else:
        df = retriever.retrieve(full_ds, topk=args.top_k_retrieval)
        df["correct"] = df["original_context"] == df["context"]
        print(
            "correct retrieval result by exhaustive search",
            df["correct"].sum() / len(df),
        )

        scores, indices = retriever.retrieve(query, topk=args.top_k_retrieval)
        
        
if __name__ == "__main__":
    parser = HfArgumentParser((DataArguments, ModelingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".yaml"):
        data_args, model_args = parser.parse_yaml_file(yaml_file=os.path.abspath(sys.argv[1]))
    else:
        data_args, model_args = parser.parse_args_into_dataclasses()
    data_args.model_name_or_path = model_args.model_name_or_path
    main(data_args)
    
    


