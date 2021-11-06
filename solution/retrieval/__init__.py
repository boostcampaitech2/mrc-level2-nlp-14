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


# @TODO: elastic search와 sparse, dense 사용법 통일하기
# ElasticSearch -> build_index
# SparseEngine -> get_passage_embeddings


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