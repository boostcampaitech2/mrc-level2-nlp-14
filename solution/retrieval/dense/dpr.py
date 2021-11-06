import os
import torch
from transformers import BertModel, BertPreTrainedModel

from solution.args import DataArguments
from .base import DenseRetrieval
from solution.utils.constant import Q_ENCODER_NAME, P_ENCODER_NAME


class BertEncoder(BertPreTrainedModel):
    """ Bert Encoder for query or passage embedding """

    def __init__(self, config):
        super(BertEncoder, self).__init__(config)

        self.bert = BertModel(config)
        self.init_weights()

    def forward(self, input_ids,
                attention_mask=None, token_type_ids=None):

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids)

        pooled_output = outputs[1]

        return pooled_output


class DensePassageRetrieval(DenseRetrieval):
    """ Dense Passage Retrieval """

    def __init__(self, args: DataArguments):
        self.tokenizer_name = args.retrieval_tokenizer_name
        self.q_encoder = torch.load(os.path.join(
            args.retrieval_model_path, f"{Q_ENCODER_NAME}.pt"))
        self.p_encoder = torch.load(os.path.join(
            args.retrieval_model_path, f"{P_ENCODER_NAME}.pt"))

        self.q_encoder.eval()
        self.p_encoder.eval()

        super().__init__(args)
        self.k = args.top_k_retrieval
        self.enable_batch = False  # batch 사용 불가
        self.args.use_faiss = False  # faiss를 사용하지 않음

    def calculate_scores(self, q_embeddings, p_embeddings):
        """ Calculate similarity scores between query and passage embeddings

        Args:
            q_embeddings (torch.Tensor): query embeddings from query encoder
            p_embeddings (torch.Tensor): passage embeddings from passage encoder

        Returns:
            torch.Tensor: similarity scores (num_queries, num_passages)
        """
        dot_prod_scores = torch.matmul(
            q_embeddings, torch.transpose(p_embeddings, 0, 1))
        return dot_prod_scores
