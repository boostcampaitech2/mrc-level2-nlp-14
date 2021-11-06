from sklearn.feature_extraction.text import TfidfVectorizer

from solution.args import MrcDataArguments
from .base import SparseRetrieval


class TfidfRetrieval(SparseRetrieval):

    def __init__(self, args: MrcDataArguments):
        self.tokenizer_name = args.retrieval_tokenizer_name
        self.vectorizer = TfidfVectorizer(
            tokenizer=self.tokenize_fn,
            ngram_range=args.sp_ngram_range,
            max_features=args.sp_max_features,
            lowercase=args.lowercase,
        )
        super().__init__(args)
        self.enable_batch = True

    def vectorize(self, contexts):
        p_embeddings = self.vectorizer.fit_transform(contexts)
        return p_embeddings

    def calculate_scores(self, q_embeddings, p_embeddings):
        return q_embeddings * p_embeddings.T
