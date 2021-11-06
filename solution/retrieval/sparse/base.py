import os
import abc
import pickle
from datasets import Dataset
from typing import Union, List, Tuple, Optional, TypeVar, Callable

from konlpy.tag import Mecab, Kkma, Okt
from transformers import AutoTokenizer

import numpy as np
from scipy.sparse.csr import csr_matrix

from solution.args import DataArguments
from solution.utils import timer
from ..core import RetrievalBase


Nested_List = List[List[int]]
Tokenizer = TypeVar("Tokenizer")


class SparseRetrieval(RetrievalBase):
    """
    Base class for Sparse Retrieval module.

    Main Method:
        get_query_embedding
        get_topk_documents

    Abstract Method
        vectorize
        calculate_scores
    """

    def __init__(self, args: DataArguments):
        super().__init__(args)
        self.name = args.retrieval_name

    @property
    def vectorizer(self):
        """ Get vectorizer object """
        return self._vectorizer

    @vectorizer.setter
    def vectorizer(self, val):
        """ Set vectorizer object """
        self._vectorizer = val

    @abc.abstractmethod
    def vectorize(self, contexts):
        """
        Perform vectorize using self.vectorizer object.
        Subclass and override for custom behavior.
        """
        pass

    @abc.abstractmethod
    def calculate_scores(self, query_embedding, passage_embedding):
        """
        Calculate similarity scores.
        Subclass and override for custom behavior.
        """
        pass

    def set_tokenizer(self) -> Tokenizer:
        """ Set tokenizer object """

        if self.tokenizer_name == "mecab":
            tokenizer = Mecab()
        elif self.tokenizer_name == "kkma":
            tokenizer = Kkma()
        elif self.tokenizer_name == "okt":
            tokenizer = Okt()
        else:
            tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        return tokenizer

    @property
    def tokenize_fn(self) -> Callable:
        """ Get tokenize function of tokenizer object """
        try:
            tokenizer = self._tokenizer
        except:
            tokenizer = self.set_tokenizer()
            self._tokenizer = tokenizer
        if self.tokenizer_name in ["mecab", "kkma", "okt"]:
            tokenize_fn = self._tokenizer.morphs
        else:
            tokenize_fn = self._tokenizer.tokenize
        return tokenize_fn

    @timer(dataset=True)
    def get_relevant_doc(
        self,
        query_or_dataset: Union[str, Dataset],
        topk: int,
        use_faiss: bool = False,
        **kwargs,
    ) -> Tuple[List, List]:
        """Get relevant document using sparse similarity function

        Args:
            query_or_dataset (Union[str, Dataset])
            topk (int): Retrieve the top k documents
            use_faiss (bool, optional): Whether to use faiss index

        Returns:
            Tuple[List, List]: [description]
        """

        return super().get_relevant_doc(query_or_dataset,
                                        topk, use_faiss, **kwargs)

    @timer(dataset=False)
    def get_query_embedding(self, query_or_dataset: Union[str, Dataset]) -> csr_matrix:
        """ Get query embedding using vectorizer objcet """

        if isinstance(query_or_dataset, Dataset):
            query = query_or_dataset["question"]
        else:
            query = [query_or_dataset]
        query_emb = self.vectorizer.transform(query)
        assert np.sum(query_emb) != 0
        return query_emb

    def get_passage_embedding(self) -> csr_matrix:
        """ Get passage embedding using vectorizer object """

        cls_name = self.__class__.__name__
        pickle_name = f"{cls_name}_embedding.bin"
        vectorizer_path = f"{cls_name}_vectorizer.bin"
        emb_path = os.path.join(self.dataset_path, pickle_name)
        vectorizer_path = os.path.join(self.dataset_path, vectorizer_path)

        if (not self.args.rebuilt_index and
            os.path.isfile(emb_path) and
                os.path.isfile(vectorizer_path)):
            with open(emb_path, "rb") as file:
                self._p_embedding = pickle.load(file)
            with open(vectorizer_path, "rb") as file:
                self._vectorizer = pickle.load(file)
            print("Embedding pickle load.")
        else:
            print("Build passage embedding")
            self._p_embedding = self.vectorize(self.contexts)
            with open(emb_path, "wb") as file:
                pickle.dump(self.p_embedding, file)
            with open(vectorizer_path, "wb") as file:
                pickle.dump(self.vectorizer, file)
            print("Embedding pickle saved.")
            self.args.rebuilt_index = False

        return self.p_embedding

    @timer(dataset=False)
    def get_topk_documents(
        self,
        query_embs: Union[csr_matrix, np.ndarray],
        topk: int,
        use_faiss: bool,
    ) -> Tuple[Nested_List, Nested_List]:
        """ Get top k document using calculate score function """
        if use_faiss:
            return self.get_topk_documents_with_faiss(query_embs, topk)
        if self.enable_batch:
            return self.get_topk_documents_bulk(query_embs, topk)
        doc_scores = []
        doc_indices = []
        for query_emb in query_embs:
            result = self.calculate_scores(query_emb, self.p_embedding)
            if not isinstance(result, np.ndarray):
                result = result.toarray()
            sorted_result = np.argsort(result)[::-1]
            doc_scores.append(result[sorted_result].tolist()[:topk])
            doc_indices.append(sorted_result.tolist()[:topk])
        return doc_scores, doc_indices

    def get_topk_documents_with_faiss(self, query_embs, topk):
        """Get top-k similarity among query and documents with faiss

        Args:
            query_emb (Union[csr_matrix, np.ndarray]):

        Returns:
            document score (List):
                topk document's similarity for input query
            document indices (List):
                topk document's indices for input query
        """
        query_embs = query_embs.toarray().astype(np.float32)
        doc_scores, doc_indices = self.indexer.search(query_embs, topk)
        return doc_scores, doc_indices

    def get_topk_documents_bulk(self, query_embs,  topk):
        """ Get top-k documents (batch version) """
        result = self.calculate_scores(query_embs, self.p_embedding)
        if not isinstance(result, np.ndarray):
            result = result.toarray()
        # batchify
        doc_scores = np.partition(result, -topk)[:, -topk:][:, ::-1]
        ind = np.argsort(doc_scores, axis=-1)[:, ::-1]
        doc_scores = np.sort(doc_scores, axis=-1)[:, ::-1].tolist()
        doc_indices = np.argpartition(result, -topk)[:, -topk:][:, ::-1]
        r, c = ind.shape
        ind += np.tile(np.arange(r).reshape(-1, 1), (1, c)) * c
        doc_indices = doc_indices.ravel()[ind].reshape(r, c).tolist()
        return doc_scores, doc_indices
