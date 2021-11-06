from solution.utils import timer
from ..core import RetrievalBase
from solution.args import DataArguments
import torch
import numpy as np
import os
import abc
import pickle
from tqdm import tqdm

from typing import Union, List, Tuple, TypeVar
from transformers import AutoTokenizer
from datasets import Dataset


Tokenizer = TypeVar("Tokenizer")
Nested_List = List[List[int]]


class DenseRetrieval(RetrievalBase):
    """ Base class for Dense Retrieval module
    Main Method:
        - get_query_embedding: Callable
        - get_passage_embedding: Callable
        - get_topk_documents: Callable
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
        if self.tokenizer_name in ["mecab", "kkma", "okt"]:
            raise ValueError("Check the tokenizer name for Dense Retrieval")
        else:
            tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        return tokenizer

    @property
    def tokenize_fn(self):
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

        return super().get_relevant_doc(query_or_dataset,
                                        topk, use_faiss, **kwargs)

    @timer(dataset=False)
    def get_query_embedding(self, query_or_dataset: Union[str, Dataset]) -> torch.Tensor:
        """ Get query embeddings """
        if isinstance(query_or_dataset, Dataset):
            query = query_or_dataset["question"]
        else:
            query = [query_or_dataset]

        with torch.no_grad():
            q_embeddings = []

            for q in query:
                q = self.tokenize_fn(
                    q, padding="max_length", truncation=True, return_tensors='pt').to('cuda')
                q_embedding = self.q_encoder(**q).to('cpu').numpy()
                q_embeddings.append(q_embedding)

        q_embeddings = torch.Tensor(
            q_embeddings).squeeze()  # (num_passage, emb_dim)
        return q_embeddings

    def get_passage_embedding(self):
        """ Get passage embeddings (load or save embeddings) """
        cls_name = self.__class__.__name__
        pickle_name = f"{cls_name}_embedding.bin"
        emb_path = os.path.join(self.dataset_path, pickle_name)

        if (not self.args.rebuilt_index and os.path.isfile(emb_path)):
            with open(emb_path, "rb") as file:
                self._p_embedding = pickle.load(file)
            print("Embedding pickle load.")
        else:
            print("Build passage embedding")
            with torch.no_grad():
                self._p_embedding = []
                batch_size = 16
                N = len(self.contexts)
                q = N // batch_size
                N_batch = batch_size * q
                for i in tqdm(range(0, N_batch, batch_size)):
                    p = self.contexts[i:i+batch_size]
                    p = self.tokenize_fn(
                        p, padding="max_length", truncation=True, return_tensors='pt').to('cuda')
                    p_emb = self.p_encoder(
                        **p).to('cpu').numpy().astype("float16")
                    self._p_embedding.append(p_emb)
                else:
                    p = self.contexts[i+batch_size:]
                    p = self.tokenize_fn(
                        p, padding="max_length", truncation=True, return_tensors='pt').to('cuda')
                    p_emb = self.p_encoder(
                        **p).to('cpu').numpy().astype("float16")
                    self._p_embedding.append(p_emb)
                self._p_embedding = np.vstack(
                    self._p_embedding)  # (num_passage, emb_dim)
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
        """ Get top-k similar documents for query

        Args:
            query_embs (Union[torch.Tensor, np.ndarray]): query embeddings
            topk (int): The number of similar documents
            use_faiss (bool): Whether to use faiss

        Returns:
            Tuple[Nested_List, Nested_List]: scores and indices of top-k documents
        """
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
