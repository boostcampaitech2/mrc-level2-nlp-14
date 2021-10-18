# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import abc
from datasets import Dataset
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModel, AutoModelForSeq2SeqLM, PreTrainedModel
from transformers.modeling_outputs import QuestionAnsweringModelOutput

class ReaderModelBase:
    """ Base class for Reader Model module """
    
    @abc.abstractmethod
    def forward(self):
        """ Call forward (fix name convention) """
        pass

    @abc.abstractmethod
    def set_trainer(self, retrieved_dataset:Dataset=None):
        """ Set up the Trainer """
        pass

    @abc.abstractmethod
    def predict(self, *args, **kwargs):
        """ Call predict """
        pass        

class ExtractiveReaderModel(nn.Module, ReaderModelBase):
    """
    HF Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear
    layer on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """,
    def __init__(
        self, backbone, input_size):
        super(ExtractiveReaderModel, self).__init__()
        self.backbone = backbone
        self.input_size = input_size


class ExtractiveReaderBaselineModel(ExtractiveReaderModel):
    """
    HF Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear
    layer on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """,
    def __init__(self, backbone, input_size):
        super(ExtractiveReaderBaselineModel, self).__init__(backbone, input_size)
        # If you want to change head layer, Overide it.
        self.qa_outputs = None if input_size is None else nn.Linear(input_size, 2)
        # backbone model's pooler output index. 2 for BERT type, 1 for the others 
        self.pooling_idx = 2 if 'bert' in self.backbone.__class__.__name__.lower() else 1

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        start_positions=None,
        end_positions=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        sequence_output = outputs[0]
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()


        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = nn.CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (start_logits, end_logits) + outputs[self.pooling_idx:]
            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

class ExtractiveReaderMLPModel(ExtractiveReaderBaselineModel):
    """ Deeper MLP Head """
    def __init__(self, input_size):
        super(ExtractiveReaderMLPModel, self).__init__(backbone, input_size)
        self.qa_outputs = None if input_size is None else nn.Sequential(
                        nn.Linear(input_size, input_size * 4, bias=False),
                        nn.Linear(input_size * 4, input_size, bias=False),
                        nn.Linear(input_size, 2))


class GenerativeReaderModel(AutoModelForSeq2SeqLM, ReaderModelBase):
    """
    HF Model with a generation head on top for generative question-answering tasks like SQuAD (a linear
    layer on top of the hidden-states output to generate `span start logits` and `span end logits`).
    """

# When you Add New Model, Append new model to this list
READER_MODEL = {'ext' : { model.__mro__[0].__name__ : model for model in
                            [
                                ExtractiveReaderBaselineModel,
                                ExtractiveReaderMLPModel,
                            ]},
                'gen' : { model.__mro__[0].__name__ : model for model in 
                            [
                                GenerativeReaderModel,
                            ]},
                }