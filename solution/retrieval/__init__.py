from .core import SearchBase
from .sparse import (
    TfidfRetrieval,
    OkapiBM25Retrieval,
)
from .dense import (
    DensePassageRetrieval,
    ColBERTRetrieval,
)
from .elastic_engine import ESRetrieval


SPARSE_RETRIEVAL = {
    "tfidf": TfidfRetrieval,
    "okapi_bm25": OkapiBM25Retrieval,
}
DENSE_RETRIEVAL = {
    "dpr": DensePassageRetrieval,
    "colbert": ColBERTRetrieval,
}
ELASTIC_ENGINE = {
    "elastic_search": ESRetrieval,
}

RETRIEVAL_HOST = {
    "sparse": SPARSE_RETRIEVAL,
    "dense": DENSE_RETRIEVAL,
    "elastic_engine": ELASTIC_ENGINE,
}
