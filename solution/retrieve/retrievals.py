from typing import Callable, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from .core import RetrievalBase, SparseRetrieval, DenseRetrieval


class TfidfRetrieval(SparseRetrieval):
    
    def __init__(self, args, tokenizer):
        self.tokenizer = tokenizer
        super().__init__(args)
    
    def tokenize_fn(self, **kwargs):
        return self.tokenizer.tokenize(**kwargs)
    
    def build_vectorizer(self):
        return TfidfVectorizer(
            tokenizer=self.tokenize_fn,
            ngram_range=self.args.sp_ngram_range,
            max_features=self.args.sp_max_features,
        )


class BM25Retrieval(SparseRetrieval):
    
    def build_vectorizer(self):
        pass


class DPRRetrieval(DenseRetrieval):
    pass


class BertRetrieval(DenseRetrieval):
    pass


class PolyEncoderRetrieval(DenseRetrieval):
    pass