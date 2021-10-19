from typing import List, Callable
from datasets import Sequence, Value, Features, Dataset, DatasetDict

from solution.args import NewTrainingArguments, DataArguments
from solution.retrieval.sparse import TfidfRetrieval
from solution.retrieval.dense import *
from solution.retrieval.elastic_engine import ESRetrieval


SPARSE_RETRIEVAL = {
    "tfidf": TfidfRetrieval,
}
DENSE_RETRIEVAL = {
    "dpr": None,
}
ELASTIC_ENGINE = {"elastic_search": ESRetrieval}

RETRIEVAL_MODE = {
    "sparse": SPARSE_RETRIEVAL,
    "dense": DENSE_RETRIEVAL,
    "elastic_engine": ELASTIC_ENGINE,
}

def run_retrieval(
    datasets: DatasetDict,
    training_args: NewTrainingArguments,
    data_args: DataArguments,
):
    retrieval_mode = RETRIEVAL_MODE[data_args.retrieval_mode]
    retriever = retrieval_mode[data_args.retrieval_name]
    df = retriever.retrieve(datasets["validation"],
                            topk=data_args.top_k_retrieval)
    if training_args.do_predict:
        f = Features(
            {
                "context": Value(dtype="string", id=None),
                "id": Value(dtype="string", id=None),
                "question": Value(dtype="string", id=None),
            }
        )
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
    datasets = DatasetDict(
        {"validation": Dataset.from_pandas(df, features=f)}
    )
    return datasets