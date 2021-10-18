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

