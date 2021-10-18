from typing import List, Optional, Tuple
from dataclasses import dataclass, field

from solution.args.argparse import lambda_field

@dataclass
class DataArguments:
    """ Arguments related to data. """
    dataset_name: str = field(
        default="./data/aistage-mrc/train_dataset", 
        metadata={"help": "The name of the dataset to use."},
    )
    dataset_path: str = field(
        default="./data/aistage-mrc",
        metadata={"help": "The path of the dataset stored"},
    )
    context_path: str = field(
        default="wikipedia_documents.json",
        metadata={"help": "File name of context documents"}
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"},
    )

    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_seq_length: int = field(
        default=384,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch (which can "
            "be faster on GPU but will be slower on TPU)."
        },
    )
    doc_stride: int = field(
        default=128,
        metadata={
            "help": "When splitting up a long document into chunks, how much stride to take between chunks."
        },
    )
    max_answer_length: int = field(
        default=30,
        metadata={
            "help": "The maximum length of an answer that can be generated. This is needed because the start "
            "and end predictions are not conditioned on one another."
        },
    )
    eval_retrieval: bool = field(
        default=True,
        metadata={"help": "Whether to run passage retrieval using sparse embedding."},
    )
    num_clusters: int = field(
        default=64,
        metadata={"help": "Define how many clusters to use for faiss."}
    )
    top_k_retrieval: int = field(
        default=1,
        metadata={
            "help": "Define how many top-k passages to retrieve based on similarity."
        },
    )
    use_faiss: bool = field(
        default=False,
        metadata={"help": "Whether to build with faiss"}
    )
    retrieval_name: bool = field(
        default="tfidf",
        metadata={"help": ""}
    )
    es_index_name: bool = field(
        default="wiki",
        metadata={"help": ""}
    )
    sp_max_features: int = field(
        default=50000,
        metadata={"help": "Max features used for TF-IDF Vectorizer."}
    )
    sp_ngram_range: Tuple[int, int] = lambda_field(
        default=(1,2),
        metadata={"help": "N-gram range used for TF-IDF Vectorizer."}
    )
    sp_tokenizer: str = field(
        default="mecab",
        metadata={"help": ""}
    )
