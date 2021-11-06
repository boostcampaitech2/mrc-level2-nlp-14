import numpy as np
import scipy.sparse as sp
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.utils.fixes import _astype_copy_false
from sklearn.utils.validation import FLOAT_DTYPES

from solution.args import MrcDataArguments
from .base import SparseRetrieval


def _document_frequency(X):
    """Count the number of non-zero values for each feature in sparse X."""
    if sp.isspmatrix_csr(X):
        return np.bincount(X.indices, minlength=X.shape[1])
    else:
        return np.diff(X.indptr)


class OkapiBM25Retrieval(SparseRetrieval):

    def __init__(self, args: MrcDataArguments):
        self.tokenizer_name = args.retrieval_tokenizer_name
        self.vectorizer = CountVectorizer(
            tokenizer=self.tokenize_fn,
            ngram_range=args.sp_ngram_range,
            max_features=args.sp_max_features,
            lowercase=args.lowercase,
        )
        super().__init__(args)
        self.b = args.b
        self.k1 = args.k1
        self.len_D = self.p_embedding.sum(1).A1
        self.avdl = self.len_D.mean()
        self.enable_batch = False  # batch 사용 불가
        self.args.use_faiss = False  # faiss를 사용하지 않음

    def vectorize(self, contexts):
        p_embeddings = self.vectorizer.fit_transform(contexts)
        self.vectorizer._idf_diag = self._calculate_idf(p_embeddings)
        return p_embeddings

    def _calculate_idf(self, X):
        # https://github.com/scikit-learn/scikit-learn/blob/844b4be24/sklearn/feature_extraction/text.py#L1609
        if not sp.issparse(X):
            X = sp.csr_matrix(X)
        return self.calculate_idf(X)

    @property
    def idf(self):
        return np.ravel(self.vectorizer._idf_diag.sum(axis=0))

    def calculate_idf(self, X):
        n_samples, n_features = X.shape
        dtype = X.dtype if X.dtype in FLOAT_DTYPES else np.float64
        df = _document_frequency(X)
        df = df.astype(dtype, **_astype_copy_false(df))
        df += 1
        n_samples += 1
        idf = np.log(n_samples / df)
        diag_idf = sp.diags(
            idf,
            offsets=0,
            shape=(n_features, n_features),
            format="csr",
            dtype=dtype,
        )
        return diag_idf

    def calculate_scores(self, q_embeddings, p_embeddings):
        b, k1, avdl = self.b, self.k1, self.avdl

        assert sp.isspmatrix_csr(q_embeddings)
        assert sp.isspmatrix_csr(p_embeddings)

        X = X.tocsc()[:, q_embeddings.indices]
        denominator = X + (k1 * (1 - b + b * self.len_D / avdl))[:, None]
        numerator = X.multiply(np.broadcast_to(self.idf, X.shape)) * (k1 + 1)
        return (numerator / denominator).sum(axis=1).A1
