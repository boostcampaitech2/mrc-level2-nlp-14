from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from solution.args import DataArguments
from solution.retrieval.sparse.base import SparseRetrieval

    
class BM25Retrieval(SparseRetrieval):
    
    def __init__(self, args: DataArguments):
        self.tokenizer_name = args.sp_tokenizer
        self.vectorizer = CountVectorizer(tokenizer=self.tokenize_fn)
        super().__init__(args)
        self.b = args.bm25_b
        self.k1 = args.bm25_k1
        self.enable_batch = False
    
    def vectorize(self, conetxts):
        p_embeddings = self.vectorizer.fit_transform(contexts)
        y = super(TfidfVectorizer, self.vectorizer).transform(X)
        self.avdl = y.sum(1).mean()
        
    # def calculate_scores(self, q_embeddings, p_embeddings):
        