from .core import RetrievalBase, SparseRetrieval, DenseRetrieval


class TfidfRetrieval(SparseRetrieval):
    pass


class BM25Retrieval(SparseRetrieval):
    pass


class DPRRetrieval(DenseRetrieval):
    pass


class BertRetrieval(DenseRetrieval):
    pass


class PolyEncoderRetrieval(DenseRetrieval):
    pass