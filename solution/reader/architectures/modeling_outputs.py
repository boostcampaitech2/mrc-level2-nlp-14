import torch
from typing import Optional, Tuple
from dataclasses import dataclass
from transformers.file_utils import ModelOutput


@dataclass
class Seq2SeqEnsembleModelOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    start_logits: torch.FloatTensor = None
    end_logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
