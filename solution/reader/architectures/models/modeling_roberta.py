import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from transformers import (
    RobertaForQuestionAnswering,
    RobertaPreTrainedModel,
    RobertaModel,
)
from transformers.modeling_outputs import QuestionAnsweringModelOutput

from ..modeling_heads import (
    QAConvSDSHead,
    QAConvHeadWithAttention,
    QAConvHead,
)


class RobertaForQA(RobertaForQuestionAnswering):
    reader_type: str = "extractive"
    
    def __init__(self, config):
        super().__init__(config)
        assert config.reader_type == self.reader_type


class RobertaForQAWithConvSDSHead(RobertaForQuestionAnswering):
    reader_type: str = "extractive"
    
    def __init__(self, config):
        # Initialize on RobertaPreTrainedModel
        super(RobertaForQuestionAnswering, self).__init__(config)
        assert config.reader_type == self.reader_type
        
        config.num_labels = 2
        self.num_labels = config.num_labels
        
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.qa_outputs = QAConvSDSHead(
            config.qa_conv_input_size,
            config.hidden_size,
            config.qa_conv_n_layers,
            config.num_labels,
        )
        
        self.init_weights()
        

class RobertaForQAWithConvHead(RobertaPreTrainedModel):
    """
    Roberta model for QA with conv head
    """
    reader_type: str = "extractive"
    
    def __init__(self, config):
        """
        Args:
            config (ModelArguments): ModelArguments
        """
        super().__init__(config)
        assert config.reader_type == self.reader_type
        assert hasattr(config, "sep_token_id")
        
        config.num_labels = 2
        self.sep_token_id = config.sep_token_id
        self.num_labels = config.num_labels
        
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        if config.model_head == "conv":
            head_cls = QAConvHead
        else:
            head_cls = QAConvHeadWithAttention
        self.qa_outputs = head_cls(config)
        
        self.init_weights()
        
    def make_token_type_ids(self, input_ids):
        token_type_ids = []
        for i, input_id in enumerate(input_ids):
            sep_idx = np.where(input_id.cpu().numpy() == self.sep_token_id)
            token_type_id = [0]*sep_idx[0][0] + [1]*(len(input_id)-sep_idx[0][0])
            token_type_ids.append(token_type_id)
        return torch.LongTensor(token_type_ids).to(input_ids.device)
        
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
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        
        token_type_ids = self.make_token_type_ids(input_ids)
        logits = self.qa_outputs(sequence_output, token_type_ids)
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

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )