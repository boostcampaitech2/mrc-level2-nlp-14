from typing import List, Optional, Tuple
from dataclasses import dataclass, field

from .base import DataArguments
from .argparse import lambda_field


@dataclass
class DataPathArguments(DataArguments):
    """ Arguments related to data. """
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
    dataset_version: str = field(
        default="v1.0.0",
        metadata={"help": "Dataset version"},
    )
    make_mask: bool = field(
        default=False,
        metadata={"help": "make masked dataset method"},
    )
    masking_type: str = field(
        default="mask",
        metadata={"help": "choose type of dataset you are going to make (mask , hard)"},
    )
    curriculum_learn: bool = field(
        default=False,
        metadata={"help": "Use curriculum learning method"},
    )
    curriculum_split_name: Optional[str] = field(
        default="./data/aistage-mrc/train_dataset",
        metadata={"help": "The name of the dataset split to use(for curriculum learning)"},
    )


@dataclass
class TokenizerArguments(DataPathArguments):
    max_seq_length: int = field(
        default=384,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. "
                "Sequences longer than this will be truncated, sequences shorter will be padded."
            )
        },
    )

    max_label_length: int = field(
        default=128,
        metadata={"help": "The maximum label length after tokenization."}
    )

    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to pad all samples to `max_seq_length`. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch "
                "(which can be faster on GPU but will be slower on TPU)."
            )
        },
    )
    doc_stride: int = field(
        default=128,
        metadata={
            "help": "When splitting up a long document into chunks, how much stride to take between chunks."
        },
    )
    return_token_type_ids: bool = field(
        default=False,
        metadata={
            "help": "Decide whether or not to return `token_type_ids` as a tokenize result."
        }
    )


@dataclass
class HighlightingArguments(TokenizerArguments):
    do_underline: bool = field(
        default=False,
        metadata={
            "help": "Whether to add underline embedding at the time of tokenizing or not"
        },
    )
    do_punctuation: bool = field(
        default=False,
        metadata={"help": "Whether to add punctuation to the context or not"},
    )
    punct_model_name_or_path: str = field(
        default="./outputs/run_test",
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        }
    )
    punct_max_seq_length: int = field(
        default=100,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    punct_use_auth_token: bool = field(
        default=False,
        metadata={"help": "Decide whether to use auth_token for puctuation."}
    )
    punct_revision: str = field(
        default="main",
        metadata={"help": "Decide which version of the model to call when performing punctuation."}
    )
    top_k_punctuation: int = field(
        default=10,
        metadata={
            "help": "Define how many top-k sentences to retrieve based on similarity."
        },
    )


@dataclass
class RetrievalArguments(HighlightingArguments):
    retrieval_mode: str = field(
        default="sparse",
        metadata={
            "help": "Decide which retrieval mode to call.",
            "choices": ["sparse", "dense", "elastic_engine"],
        }
    )
    retrieval_name: str = field(
        default="tfidf",
        metadata={
            "help": "Decide which retrieval class to call.",
            "choices": ["tfidf", "okapi_bm25", "dpr", "colbert", "elastic_search"],
        }
    )
    rebuilt_index: bool = field(
        default=False,
        metadata={"help": "Decide whether to rebuild search engine indexes."}
    )
    retrieval_tokenizer_name: str = field(
        default="mecab",
        metadata={"help": "Decide which search engine tokenizer to use."}
    )
    retrieval_model_path: str = field(
        default="./dense_retrieval",
        metadata={"help": "Specifies the path to the retrieval model."}
    )
    sp_max_features: int = field(
        default=50000,
        metadata={"help": "Max features used for TF-IDF Vectorizer."}
    )
    sp_ngram_range: List[int] = lambda_field(
        default=[1, 2],
        metadata={"help": "N-gram range used for TF-IDF Vectorizer."}
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
    eval_retrieval: bool = field(
        default=True,
        metadata={
            "help": "Whether to run passage retrieval using sparse embedding."
        },
    )
    num_clusters: int = field(
        default=64,
        metadata={"help": "Define how many clusters to use for faiss."}
    )


@dataclass
class ElasticSearchArguments(RetrievalArguments):
    index_name: str = field(
        default="wiki-index",
        metadata={"help": "The name of the index to use in Elasticsearch."}
    )
    stopword_path: str = field(
        default="user_dic/my_stop_dic.txt",
        metadata={"help": "Path of stopword to use in Elasticsearch."}
    )
    decompound_mode: str = field(
        default="mixed",
        metadata={
            "help": "Determines how the tokenizer handles compound tokens."
            "https://www.elastic.co/guide/en/elasticsearch/plugins/current/analysis-nori-tokenizer.html"
        }
    )
    b: float = field(
        default=0.75,
        metadata={
            "help": "Controls to what degree document length normalizes tf values. "
            "The default value is 0.75"
        }
    )
    k1: float = field(
        default=1.2,
        metadata={
            "help": "Controls non-linear term frequency normalization (saturation). "
            "The default value is 1.2."
        }
    )
    es_host_address: str = field(
        default="localhost:9200",
        metadata={"help": "Network host address"}
    )
    es_timeout: int = field(
        default=30,
        metadata={"help": "Determine connection timeout thershold."}
    )
    es_max_retries: int = field(
        default=10,
        metadata={"help": "Determine connection timeout thershold."}
    )
    es_retry_on_timeout: bool = field(
        default=True,
        metadata={"help": "Specifies the maximum number of connection attempts."}
    )
    es_similarity: str = field(
        default="bm25_similarity",
        metadata={"help": "Decide which similarity calculation to use."}
    )
    use_korean_stopwords: bool = field(
        default=False,
        metadata={"help": "Decide whether to use the Korean stopword dictionary provided by Elastic Search."}
    )
    use_korean_synonyms: bool = field(
        default=False,
        metadata={"help": "Decide whether to use the Korean synonym dictionary provided by Elastic Search."}
    )
    lowercase: bool = field(
        default=False,
        metadata={"help": "Determines whether text is treated as lowercase."}
    )
    nori_readingform: bool = field(
        default=False,
        metadata={"help": "Filter rewrites tokens written in Hanja to their Hangul form."}
    )
    cjk_bigram: bool = field(
        default=False,
        metadata={"help": "Determines whether to use bigram for Chinese, English and Korean."}
    )
    decimal_digit: bool = field(
        default=False,
        metadata={"help": "Filter folds unicode digits to 0-9"}
    )
    dfr_basic_model: str = field(
        default="g",
        metadata={
            "help": "Basic model of information content for `divergence from randomness`."
            "choices": ["g", "if", "in", "ine"],
        }
    )
    dfr_after_effect: str = field(
        default="l",
        metadata={
            "help": "First normalization of information gain.",
            "choices": ["b", "l"],
        }
    )
    es_normalization: str = field(
        default="h2",
        metadata={
            "help": "Second (length) normalization",
            "choices": ["no", "h1", "h2", "h3", "z"],
        }
    )
    dfi_measure: str = field(
        default="standardized",
        metadata={
            "help": "Three basic measures of divergence from independence",
            "choices": ["standardized", "saturated", "chisquared"],
        },
    )
    ib_distribution: str = field(
        default="ll",
        metadata={
            "help": "Probabilistic distribution used to model term occurrence",
            "choices": ["ll", "spl"],
        }
    )
    ib_lambda: str = field(
        default="df",
        metadata={
            "help": ":math:`λ_w` parameter of the probability distribution",
            "choices": ["df", "ttf"],
        }
    )
    lmd_mu: int = field(
        default=2000,
        metadata={"help": "Parameters to be used in `LM Dirichlet similarity`."}
    )
    lmjm_lambda: float = field(
        default=0.1,
        metadata={
            "help": (
                "The optimal value depends on both the collection and the query. "
                "The optimal value is around 0.1 for title queries and 0.7 for long queries. "
                "Default to 0.1. When value approaches 0, "
                "documents that match more query terms will be ranked higher than those that match fewer terms."
            )
        }
    )


@dataclass
class DenoisingArguments(ElasticSearchArguments):
    denoising_func: Optional[str] = field(
        default=None,
        metadata={"help": "Decide which denoising function to use."}
    )
    permute_sentence_ratio: float = field(
        default=1.0,
        metadata={
            "help": "Decide how much to shuffle sentences.",
            "choices": range(0, 1),
        }
    )


@dataclass
class MrcDataArguments(DenoisingArguments):
    pass
