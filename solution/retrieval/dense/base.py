import os
import abc
import pickle
from tqdm import tqdm

from typing import Union, List, Tuple, TypeVar
from transformers import AutoTokenizer
from datasets import Dataset


Tokenizer = TypeVar("Tokenizer")
Nested_List = List[List[int]]

import numpy as np
import torch

from solution.args import DataArguments
from ..core import RetrievalBase
from solution.utils import timer


class DenseRetrieval(RetrievalBase):
    """ Base class for Dense Retrieval module
    Main Method:
        - get_query_embedding: Callable
        - get_passage_embedding: Callable
        - get_topk_documents: Callable
    
    Abstract Method:
        - train: Callable
        - load_model: Callable
        - encode: (vectorize)
        
    Attributes:
        - p_embedding: 삭제
        - encoder
        - use_faiss: bool
    
    """
    def __init__(self, args: DataArguments):
        super().__init__(args)
        self.name = args.retrieval_name
        
    @property
    def q_encoder(self):
        return self._q_encoder
    
    @q_encoder.setter
    def q_encoder(self, val):
        self._q_encoder = val
    
    @property
    def p_encoder(self):
        return self._p_encoder
    
    @p_encoder.setter
    def p_encoder(self, val):
        self._p_encoder = val
    
    @abc.abstractmethod
    def calculate_scores(self, query_embedding, passage_embedding):
        pass
    
    def set_tokenizer(self) -> Tokenizer:
        # Argument 구성에 따라 수정될 수 있음
        if self.tokenizer_name in ["mecab", "kkma", "okt"]:
            raise ValueError("Check the tokenizer name for Dense Retrieval")
        else:
            tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        return tokenizer
    
    @property
    def tokenize_fn(self):
        # 삭제할 가능성 있음 (tokenizer 자체로 call 가능)
        try:
            tokenizer = self._tokenizer
        except:
            tokenizer = self.set_tokenizer()
            self._tokenizer = tokenizer
        tokenize_fn = self._tokenizer
        
        return tokenize_fn
    
    @timer(dataset=True)
    def get_relevant_doc(
        self,
        query_or_dataset: Union[str, Dataset],
        topk: int,
        use_faiss: bool = False,
        **kwargs,
    ) -> Tuple[List, List]:
        
        # get_query_embedding, get_topk_documents가 Dense에 맞게 수정되면 수정 필요
        return super().get_relevant_doc(query_or_dataset,
                                        topk, use_faiss, **kwargs)
    
    @timer(dataset=False)
    def get_query_embedding(self, query_or_dataset: Union[str, Dataset]) -> torch.Tensor:
        if isinstance(query_or_dataset, Dataset):
            query = query_or_dataset["question"]
        else:
            query = [query_or_dataset]
            
        with torch.no_grad():
            #self.q_encoder.eval()
            q_embeddings = []

            for q in query:
                q = self.tokenize_fn(q, padding="max_length", truncation=True, return_tensors='pt').to('cuda')
                q_embedding = self.q_encoder(**q).to('cpu').numpy()
                q_embeddings.append(q_embedding)
        
        q_embeddings = torch.Tensor(q_embeddings).squeeze()  # (num_passage, emb_dim)
        return q_embeddings

    
    def get_passage_embedding(self):
        # q_encoder, p_encoder 학습 되어있음을 가정
        cls_name = self.__class__.__name__
        pickle_name = f"{cls_name}_embedding.bin"
        emb_path = os.path.join(self.dataset_path, pickle_name)
        # 저장된 파일 있으면 불러옴
        if (not self.args.rebuilt_index and os.path.isfile(emb_path)):
            with open(emb_path, "rb") as file:
                self._p_embedding = pickle.load(file)
            print("Embedding pickle load.")
        # 없으면 만드는데
        else:
            print("Build passage embedding")
            
            with torch.no_grad():
                #self.p_encoder.eval()
                self._p_embedding = []
                batch_size = 16
                N = len(self.contexts)
                q = N // batch_size
                N_batch = batch_size * q
                for i in tqdm(range(0, N_batch, batch_size)):
                    p = self.contexts[i:i+batch_size]
                    p = self.tokenize_fn(p, padding="max_length", truncation=True, return_tensors='pt').to('cuda')
                    p_emb = self.p_encoder(**p).to('cpu').numpy().astype("float16")
                    self._p_embedding.append(p_emb)
                self._p_embedding = np.vstack(self._p_embedding)  # (num_passage, emb_dim)
                self._p_embedding = torch.from_numpy(self._p_embedding)

            with open(emb_path, "wb") as file:
                pickle.dump(self.p_embedding, file)
            print("Embedding pickle saved.")
            self.args.rebuilt_index = False
                
        return self.p_embedding
    
    @timer(dataset=False)
    def get_topk_documents(
        self, 
        query_embs: Union[torch.Tensor, np.ndarray], 
        topk: int,
        use_faiss: bool,
    ) -> Tuple[Nested_List, Nested_List]:
        # @TODO: faiss, batch 구현
        #if use_faiss:
        #    return self.get_topk_documents_with_faiss(query_embs, topk)
        #if self.enable_batch:
        #    return self.get_topk_documents_bulk(query_embs, topk)
        doc_scores = []
        doc_indices = []
        for query_emb in tqdm(query_embs):
            result = self.calculate_scores(query_emb, self.p_embedding)
            if not isinstance(result, np.ndarray):
                result = result.numpy()
            sorted_result = np.argsort(result)[::-1]
            doc_scores.append(result[sorted_result].tolist()[:topk])
            doc_indices.append(sorted_result.tolist()[:topk])
        return doc_scores, doc_indices

    def get_topk_documents_with_faiss(self, query_embs, topk):
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
        query_embs = query_embs.toarray().astype(np.float32)
        doc_scores, doc_indices = self.indexer.search(query_embs, topk)
        return doc_scores, doc_indices
    
    def get_topk_documents_bulk(self, query_embs,  topk):
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
