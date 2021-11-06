import os
import string
import torch
import torch.nn as nn

from transformers import BertPreTrainedModel, BertModel, BertTokenizerFast

from solution.args import DataArguments
from .base import DenseRetrieval
from solution.utils.constant import COLBERT_NAME


class ColBERTEncoder(BertPreTrainedModel):
    """ ColBERT Encoder for query or passage embedding """

    def __init__(self, config, mask_punctuation, dim=128, similarity_metric="cosine"):
        super(ColBERTEncoder, self).__init__(config)

        self.similarity_metric = similarity_metric
        self.dim = dim

        self.mask_punctuation = mask_punctuation
        self.skiplist = {}

        if self.mask_punctuation:
            self.tokenizer = BertTokenizerFast.from_pretrained(
                "klue/bert-base")
            self.skiplist = {
                w: True
                for symbol in string.punctuation
                for w in [symbol, self.tokenizer.encode(symbol, add_special_tokens=False)[0]]
            }

        self.bert = BertModel(config)
        self.linear = nn.Linear(config.hidden_size, dim, bias=False)

        self.init_weights()

    def forward(self, Q, D):
        """ Return similarity score between Q and D using score function """
        return self.score(Q, D)

    def query(self, input_ids, attention_mask, token_type_ids=None):
        """ Return query embeddings """
        input_ids, attention_mask = input_ids.to(
            "cuda"), attention_mask.to("cuda")
        Q = self.bert(input_ids, attention_mask=attention_mask)[0]
        Q = self.linear(Q)

        return torch.nn.functional.normalize(Q, p=2, dim=2)

    def doc(self, input_ids, attention_mask, token_type_ids=None, keep_dims=True):
        """ Return document(passage) embeddings """
        input_ids, attention_mask = input_ids.to(
            "cuda"), attention_mask.to("cuda")
        D = self.bert(input_ids, attention_mask=attention_mask)[0]
        D = self.linear(D)

        mask = torch.tensor(self.mask(input_ids),
                            device="cuda").unsqueeze(2).float()
        D = D * mask

        D = torch.nn.functional.normalize(D, p=2, dim=2)

        if not keep_dims:
            D, mask = D.cpu().to(dtype=torch.float16), mask.cpu().bool().squeeze(-1)
            D = [d[mask[idx]] for idx, d in enumerate(D)]

        return D

    def score(self, Q, D):
        """ Return similarity score between Q and D

        Args:
            Q (torch.Tensor): query embeddings
            D (torch.Tensor): document(passage) embeddings

        Returns:
            torch.Tensor: similarity score
        """
        if self.similarity_metric == "cosine":
            return (Q.float() @ D.float().permute(0, 2, 1)).max(2).values.sum(1)

        assert self.similarity_metric == "l2"
        return (-1.0 * ((Q.unsqueeze(2) - D.unsqueeze(1)) ** 2).sum(-1)).max(-1).values.sum(-1)

    def mask(self, input_ids):
        mask = [[(x not in self.skiplist) and (x != 0) for x in d]
                for d in input_ids.cpu().tolist()]
        return mask


class ColBERTRetrieval(DenseRetrieval):
    """ ColBERT Retrieval """

    def __init__(self, args: DataArguments):
        self.tokenizer_name = args.retrieval_tokenizer_name
        self.colbert = torch.load(os.path.join(
            args.retrieval_model_path, f"{COLBERT_NAME}.pt"))
        self.colbert.eval()
        self.q_encoder = self.colbert.query
        self.p_encoder = self.colbert.doc

        super().__init__(args)
        self.k = args.top_k_retrieval
        self.enable_batch = False  # batch 사용 불가
        self.args.use_faiss = False  # faiss를 사용하지 않음

    def calculate_scores(self, q_embeddings, p_embeddings):
        """ Calculate similarity scores between query and passage embeddings

        Args:
            q_embeddings (torch.Tensor): query embeddings from ColBERT
            p_embeddings (torch.Tensor): passage embeddings from ColBERT

        Returns:
            torch.Tensor: similarity scores (num_queries, num_passages)
        """
        batch_size = 16
        N = len(p_embeddings)
        q = N // batch_size
        N_batch = batch_size * q
        scores = []
        for i in range(0, N_batch, batch_size):
            p_emb = p_embeddings[i:i+batch_size]
            score = self.colbert(q_embeddings.unsqueeze(dim=0), p_emb)
            scores.append(score)
        else:
            p_emb = p_embeddings[i+batch_size:]
            score = self.colbert(q_embeddings.unsqueeze(dim=0), p_emb)
            scores.append(score)
        return torch.cat(scores)
