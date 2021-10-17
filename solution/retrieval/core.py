import os
import abc
import time
import json
import pickle
from typing import List, Union, Tuple, Optional

import scipy
from scipy.sparse.csr import csr_matrix
import numpy as np
import pandas as pd
from datasets import Dataset

from ..args import DataArguments
from .retrieve_mixin import FaissMixin, PandasMixin


Nested_List = List[List[int]]


def timer(dataset=True):
    def decorator(func):
        """ Time decorator """
        flag = True if dataset else False
        def wrap_func(self, query_or_dataset, *args, **kwargs):
            dataset_cond = isinstance(query_or_dataset, Dataset) and flag
            str_cond = isinstance(query_or_dataset, str) and not flag
            if dataset_cond or str_cond:
                t0 = time.time()
            output = func(self, query_or_dataset, *args, **kwargs)
            if dataset_cond or str_cond:
                print(f"[{func.__name__}] done in {time.time() - t0:.3f} s")
            return output
        return wrap_func
    return decorator


class RetrievalBase(FaissMixin, PandasMixin):
    """ Base class for Retrieval module """
    
    @abc.abstractproperty
    def contexts(self):
        """ Get corpus contexts (fix name convention) """
        pass
    
    @abc.abstractproperty
    def p_embedding(self):
        """ Get passage embeddings (fix name convention) """
        pass
    
    @abc.abstractproperty
    def use_faiss(self):
        """ Whether to use faiss or not """
        pass
    
    def retrieve(
        self, 
        query_or_dataset: Union[str, Dataset], 
        topk: Optional[int] = 1,
        **kwargs,
    ) -> Union[Tuple[List, List], pd.DataFrame]:
        """
        Arguments:
            query_or_dataset (Union[str, Dataset]):
                str이나 Dataset으로 이루어진 Query를 받습니다.
            topk (Optional[int], optional): Defaults to 1.
                상위 몇 개의 passage를 사용할 것인지 지정합니다.
        Returns:
            1개의 Query를 받는 경우  -> Tuple(List, List)
            다수의 Query를 받는 경우 -> pd.DataFrame: [description]
        Note:
            다수의 Query를 받는 경우,
                Ground Truth가 있는 Query (train/valid) -> 기존 Ground Truth Passage를 같이 반환합니다.
                Ground Truth가 없는 Query (test) -> Retrieval한 Passage만 반환합니다.
        """
        doc_scores, doc_indices = self.get_relevant_doc(query_or_dataset,
                                                        k=topk,
                                                        use_faiss=self.args.use_faiss,
                                                        **kwargs)
        if isinstance(query_or_dataset, str):
            doc_scores = doc_scores[0]
            doc_indices = doc_indices[0]
            print("[Search query]\n", query_or_dataset, "\n")
            
            for i in range(topk):
                print("Top-%d passage with score %.4f" % (i + 1, doc_scores[i]))
                print(self.contexts[doc_indices[i]], end="\n\n")

            return (doc_scores, [self.contexts[doc_indices[i]] for i in range(topk)])
        
        elif isinstance(query_or_dataset, Dataset):
            cqas = self.get_dataframe_result(query_or_dataset, doc_scores, doc_indices)
            return cqas
    
    def get_relevant_doc(
        self,
        query_or_dataset: Union[str, Dataset],
        k: int,
        use_faiss: bool = False,
        **kwargs,
    ) -> Tuple[List, List]:
        """
        입력 query에 관련이 있는 상위 k의 document를 검색
        
        Arguments:
            query_or_dataset (Union[str, Dataset]):
                하나의 Query 혹은 HF.Dataset를 입력으로 받음
            k (int):
                상위 몇 개의 pasaage를 반환할지 결정
        Returns:
            document score (List):
                입력 query에 대한 topk document 유사도
            document indices (List):
                입력 query에 대한 topk document 인덱스
        """
        query_emb = self.get_query_embedding(query_or_dataset, **kwargs)
        doc_scores, doc_indices = self.get_topk_similarity(query_emb, k, use_faiss, **kwargs)
        return doc_scores, doc_indices
        
    @abc.abstractmethod
    def get_query_embedding(self, query, **kwargs):
        """ Get query embedding """
        pass
    
    @abc.abstractmethod
    def get_topk_similarity(self, query_emb, k, use_faiss, **kwargs):
        """ Get top-k similarity among query and documents """
        pass
        

class SparseRetrieval(RetrievalBase):
    """ Base class for Sparse Retrieval module """
    
    def __init__(self, args: DataArguments):
        self.args = args
        self.data_path = args.dataset_path
        self.context_filename = args.context_path
        with open(os.path.join(self.data_path, args.context_path), "r", encoding="utf-8") as f:
            corpus = json.load(f)
        self._contexts = list(
            dict.fromkeys([v["text"] for v in corpus.values()])
        )
        print(f"Lengths of unique contexts : {len(self.contexts)}")
        self.ids = list(range(len(self._contexts)))
        self._p_embedding = None        
        self._build_vectorizer()
        
        # initialize passage embedding and fit the model
        _ = self.p_embedding
        
        if args.use_faiss:
            self.build_faiss(args.data_path, args.num_clusters)
            
    @property
    def use_faiss(self):
        """ Whether to use faiss or not """
        return self.args.use_faiss
        
    @property
    def vectorizer(self):
        """
        Get vectorizer to vectorize query and passages.
        If self._vectorizer is None, raise AttributeError.
        """
        vectorizer = self._vectorizer
        if vectorizer is None:
            raise AttributeError("There is no vectorizer. "
                                 "Implement `build_vectorizer` method "
                                 "and run it in the constructor.")
        return vectorizer
    
    @vectorizer.setter
    def vectorizer(self, vec_object):
        """ Set vectorizer to vectorize query and passages. """
        if not all(hasattr(vec_object, attr) for attr in ["fit", "fit_transform", "transform"]):
            raise AttributeError("This vectorizer hasn't `fit`, `fit_transform`, `transform` method.")
        self._vectorizer = vec_object
        
    @property
    def contexts(self):
        """ Get corpus contexts (fix name convention) """
        return self._contexts
    
    @property
    def p_embedding(self):
        """
        Get passage embeddings (fix name convention)
        If self._p_embedding is None,
        run self.get_sparse_embedding method to get passage embedding.
        """
        p_embedding = self._p_embedding
        if p_embedding is None:
            p_embedding = self.get_sparse_embedding()
            self._p_embedding = p_embedding
        return p_embedding
    
    @abc.abstractmethod
    def build_vectorizer(self):
        """
        Build user-used vectorizer.
        You must implement this method to get vectorizer.
        """
        pass
    
    def _build_vectorizer(self):
        """ private method for build vectorizer method. """
        self.vectorizer = self.build_vectorizer()
    
    def get_sparse_embedding(self):
        """
        Passage Embedding을 만들고
        TFIDF와 Embedding을 pickle로 저장합니다.
        만약 미리 저장된 파일이 있으면 저장된 pickle을 불러옵니다.
        """
        pickle_name = f"sparse_embedding.bin"
        tfidfv_name = f"tfidv.bin"
        emb_path = os.path.join(self.data_path, pickle_name)
        tfidfv_path = os.path.join(self.data_path, tfidfv_name)
        
        if os.path.isfile(emb_path) and os.path.isfile(tfidfv_path):
            with open(emb_path, "rb") as file:
                p_embedding = pickle.load(file)
            with open(tfidfv_path, "rb") as file:
                self.vectorizer = pickle.load(file)
            print("Embedding pickle load.")
        else:
            print("Build passage embedding")
            p_embedding = self.vectorizer.fit_transform(self.contexts)
            with open(emb_path, "wb") as file:
                pickle.dump(self.p_embedding, file)
            with open(tfidfv_path, "wb") as file:
                pickle.dump(self.tfidfv, file)
            print("Embedding pickle saved.")
                
        return p_embedding
    
    @timer(dataset=True)
    def get_relevant_doc(
        self,
        query_or_dataset: Union[str, Dataset],
        k: int,
        use_faiss: bool = False,
        **kwargs,
    ) -> Tuple[List, List]:
        return super().get_relevant_doc(query_or_dataset,
                                        k, use_faiss, **kwargs)
    
    @timer(dataset=False)
    def get_query_embedding(self, query_or_dataset: Union[str, Dataset]) -> csr_matrix:
        """
        Get query embedding using vectorizer.
        
        Arguments:
            query_or_dataset (Union[str, Dataset]):
                하나의 Query 혹은 HF.Dataset를 입력으로 받음
        Returns:
            query_emb (scipy.sparse.csr.csr_matrix):
                query embedding
        """
        if isinstance(query_or_dataset, Dataset):
            query = query_or_dataset["question"]
        else:
            query = [query_or_dataset]
        query_emb = self.vectorizer.transform(query)
        assert np.sum(query_emb) != 0
        return query_emb
    
    @timer(dataset=False)
    def get_topk_similarity(
        self, 
        query_emb: Union[csr_matrix, np.ndarray], 
        k: int,
        use_faiss: bool,
    ) -> Tuple[Nested_List, Nested_List]:
        """
        Get top-k similarity among query and documents
        
        Arguments:
            query_emb (Union[csr_matrix, np.ndarray]):
        Returns:
            document score (List):
                입력 query에 대한 topk document 유사도
            document indices (List):
                입력 query에 대한 topk document 인덱스
        """
        if use_faiss:
            docs_scores, doc_indices = self.get_similarity_with_faiss(query_emb)
        else:
            result = query_emb * self.p_embedding.T
            if not isinstance(result, np.ndarray):
                result = result.toarray()
            # batchify
            doc_scores = np.partition(result, -k)[:, -k:][:, ::-1]
            ind = np.argsort(doc_scores, axis=-1)[:, ::-1]
            doc_scores = np.sort(doc_scores, axis=-1)[:, ::-1].tolist()
            doc_indices = np.argpartition(result, -k)[:, -k:][:, ::-1]
            r, c = ind.shape
            ind += np.tile(np.arange(r).reshape(-1, 1), (1, c)) * c
            doc_indices = doc_indices.ravel()[ind].reshape(r, c).tolist()
        return doc_scores, doc_indices
        
    def get_topk_similarity_with_faiss(self, query_emb, k):
        """
        Get top-k similarity among query and documents with faiss
        
        Arguments:
            query_emb (Union[csr_matrix, np.ndarray]):
        Returns:
            document score (List):
                입력 query에 대한 topk document 유사도
            document indices (List):
                입력 query에 대한 topk document 인덱스
        """
        query_emb = query_emb.toarray().astype(np.float32)
        doc_scores, doc_indices = self.indexer.search(query_emb, k)
        return doc_scores, doc_indices


class DenseRetrieval(RetrievalBase):
    """ Base class for Dense Retrieval module """
    pass
