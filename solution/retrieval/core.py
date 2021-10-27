import os
import abc
import json
from typing import List, Union, Tuple, Optional, Callable, Any

from scipy.sparse.csr import csr_matrix
import numpy as np
import pandas as pd
from datasets import Sequence, Value, Features, Dataset, DatasetDict

from solution.args import DataArguments
from solution.retrieval.retrieve_mixin import FaissMixin, PandasMixin


ArrayMatrix = Union[csr_matrix, np.ndarray]


class SearchBase(PandasMixin):
    """
    Base class for Search Engine.
    
    Abstract method:
        - retrieve: Callable
        - get_relevant_doc: Callable
    
    Attributes:
        - contexts: List[str]
        - contexts_ids: List[int]
        - context_file_path: str
        - dataset_path: str
    """
    
    def __init__(self, args: DataArguments):
        self.args = args
        with open(os.path.join(args.dataset_path, args.context_path), "r", encoding="utf-8") as f:
            corpus = json.load(f)
        self._contexts = list(
            dict.fromkeys([v["text"] for v in corpus.values()])
        )
        print(f"Lengths of unique contexts : {len(self.contexts)}")
        self._context_ids = list(
            dict.fromkeys([v["document_id"] for v in corpus.values()])
        )
        
    @property
    def contexts(self) -> List[str]:
        """
        Get corpus contexts(fix name convention).
        When the object is created, contexts are read from the corpus
        ans assigned as attribute.
        """
        return self._contexts
    
    @property
    def contexts_ids(self) -> List[int]:
        """
        Get corpus contexts ids(fix name convention).
        When the object is created, contexts are read from the corpus
        ans assigned as attribute.
        """
        return self._context_ids 
    
    @property
    def context_file_path(self) -> str:
        """ Get context file path. """
        return os.path.join(self.args.dataset_path, self.args.context_path)
    
    @property
    def dataset_path(self) -> str:
        """ Get context data path for caching. """
        return self.args.dataset_path
    
    @abc.abstractmethod
    def retrieve(self, query, topk, **kwargs) -> Any:
        pass
    
    @abc.abstractmethod
    def get_relevant_doc(self, query, topk, **kwargs) -> Tuple[List, List]:
        pass
    

class RetrievalBase(SearchBase, FaissMixin):
    """
    Base class for Retrieval module.
    
    Main method:
        - retrieve: Callable
        - get_relevant_doc: Callable
        
    Abstract method:
        - get_query_embedding: Callable
        - get_passage_embedding: Callable
        - get_topk_documents: Callable
        
    Attributes:
        - p_embedding: ArrayMatrix
        - use_faiss: bool
    """
    
    def __init__(self, args: DataArguments):
        super().__init__(args)
        
        self.get_passage_embedding()
        
        if args.use_faiss:
            self.build_faiss(args.dataset_path, args.num_clusters)
    
    @property
    def p_embedding(self) -> ArrayMatrix:
        """
        Get passage embeddings(fix name convention).
        When the object is created, execute the `get_passage_embedding` method
        to get passage embedding from the context attribute.
        """
        return self._p_embedding
    
    @property
    def use_faiss(self) -> bool:
        """ Whether to use faiss or not """
        return self.args.use_faiss
    
    @abc.abstractmethod
    def get_query_embedding(self, query, **kwargs):
        """
        Get query embedding.
        This method is called dynamically 
        when the `get_relevant_doc` method is executed.
        """
        pass
    
    @abc.abstractmethod
    def get_passage_embedding(self, passage, **kwargs):
        """
        Get passage embedding.
        This method is executed when the object is created.
        For efficient retrieval, the cache file is stored
        in the `context_file_path` at the first call.
        """
        pass
    
    @abc.abstractmethod
    def get_topk_documents(self, query_embs, topk, use_faiss, **kwargs):
        """
        Get top-k documents and ids among query and documents.
        
        Follow the steps below.
            1. Calculate the similarity among query and passage embedding
               based on the given similarity function(`calculate_scores`).
            2. Returns the top-k documents with the highest similarity and their index.
            If you use faiss library, faiss does the second job for you.
        """
        pass
    
    def retrieve(
        self, 
        query_or_dataset: Union[str, Dataset], 
        topk: Optional[int] = 1,
        **kwargs,
    ) -> Union[Tuple[List, List], pd.DataFrame]:
        """
        Retrieves the top k most similar documents from the input query
        and returns them as `DatasetDict` objects in the huggingface Datasets.
        
        Arguments:
            query_or_dataset: Union[str, Dataset]
                use bulk or not
            topk: Optional[int] Defaults to 1
        Returns:
            If type of query_or_dataset is string:
                Tuple[List, List]
                    first element: document scores
                    second element: passage corresponding to scores
            otherwise:
                pd.DataFrame
        """
        doc_scores, doc_indices = self.get_relevant_doc(query_or_dataset,
                                                        topk=topk,
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
        topk: int,
        use_faiss: bool = False,
        **kwargs,
    ) -> Tuple[List, List]:
        """
        Retrieve top k documents related to the input query
        
        Arguments:
            query_or_dataset: Union[str, Dataset]
                use bulk or not
            topk: Optional[int] Defaults to 1
            use_faiss: bool
        Returns:
            document score: List[int]
                입력 query에 대한 topk document 유사도
            document indices: List[str]
                입력 query에 대한 topk document 인덱스
        """
        query_embs = self.get_query_embedding(query_or_dataset, **kwargs)
        doc_scores, doc_indices = self.get_topk_documents(query_embs, topk, use_faiss, **kwargs)
        return doc_scores, doc_indices