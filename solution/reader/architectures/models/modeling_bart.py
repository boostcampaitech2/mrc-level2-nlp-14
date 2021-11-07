import torch
from torch import nn
from torch.nn import CrossEntropyLoss

from transformers import (
    BartForQuestionAnswering as QA,
    BartForConditionalGeneration as CG,
    BartModel,
)

from ..modeling_heads import QAConvSDSHead
from ..modeling_outputs import Seq2SeqEnsembleModelOutput


class BartForQuestionAnswering(QA):
    reader_type: str = "extractive"

    def __init__(self, config):
        super().__init__(config)
        assert config.reader_type == self.reader_type


class BartForConditionalGeneration(CG):
    reader_type: str = "generative"

    def __init__(self, config):
        super().__init__(config)
        assert config.reader_type == self.reader_type


class BartForQAWithConvSDSHead(QA):
    """
    Bart model for QA with SDS conv head
    """
    reader_type: str = "extractive"

    def __init__(self, config):
        # Initialize on BartPretrainedModel
        super(BartForQuestionAnswering, self).__init__(config)
        assert config.reader_type == self.reader_type

        config.num_labels = 2
        self.num_labels = config.num_labels

        self.model = BartModel(config)
        self.qa_outputs = QAConvSDSHead(
            config.qa_conv_input_size,
            config.hidden_size,
            config.qa_conv_n_layers,
            config.num_labels,
        )

        self.model._init_weights(self.qa_outputs)


class BartForExtractionGenerationEnsemble(CG):
    """
    Bart model for Ensemble of QA and CG
    """
    reader_type: str = "ensemble"

    def __init__(self, config):
        super().__init__(config)
        assert config.reader_type == self.reader_type

        config.num_labels = 2
        self.num_labels = config.num_labels
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)
        self.model._init_weights(self.qa_outputs)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        start_positions=None,
        end_positions=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        return_dict=None,
    ):

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if start_positions is not None and end_positions is not None:
            use_cache = False

        if labels is not None:
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            encoder_outputs=encoder_outputs,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=return_dict,
        )

        sequence_output = outputs.encoder_last_hidden_state

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        total_loss = None
        if start_positions is not None and end_positions is not None:
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        lm_logits = self.lm_head(
            outputs.last_hidden_state) + self.final_logits_bias

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(
                lm_logits.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            ext_output = (
                start_logits,
                end_logits,
            ) + outputs[1:]
            gen_output = (lm_logits,) + outputs[1:]
            return ((total_loss + masked_lm_loss,) + ext_output + gen_output) if total_loss is not None and masked_lm_loss is not None else ext_output + gen_output

        loss = None
        if total_loss is not None and masked_lm_loss is not None:
            loss = total_loss + masked_lm_loss

        return Seq2SeqEnsembleModelOutput(
            loss=loss,
            logits=lm_logits,
            start_logits=start_logits,
            end_logits=end_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )


def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id

    assert pad_token_id is not None, "self.model.config.pad_token_id has to be defined."
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids
