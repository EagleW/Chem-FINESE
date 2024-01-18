import random
import numpy as np
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
from transformers import BartTokenizer


import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss, MarginRankingLoss
import torch.nn.functional as F

from transformers.modeling_outputs import ModelOutput, BaseModelOutputWithPastAndCrossAttentions, BaseModelOutput, Seq2SeqModelOutput, Seq2SeqLMOutput
from transformers.utils import logging

from transformers.models.bart.modeling_bart import (
    BartAttention,
    BartEncoderLayer,
    BartDecoderLayer, 
    BartEncoder, 
    BartDecoder,
    BartModel, 
    BartForConditionalGeneration,
    BartPretrainedModel,
    BartConfig, 
    BartLearnedPositionalEmbedding,
    shift_tokens_right, _expand_mask, ACT2FN
)
import math
import pickle
from dataclasses import dataclass


@dataclass
class NewBaseModelOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class ContrastiveSeq2SeqLMOutput(ModelOutput):

    loss: Optional[torch.FloatTensor] = None
    mlm_loss: Optional[torch.FloatTensor] = None
    cl_loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
import torch.nn.functional as F
logger = logging.get_logger(__name__)


class ContrastiveHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(
        self,
        inner_dim: int,
        pooler_dropout: float,
    ):
        super().__init__()
        self.out_proj = nn.Linear(inner_dim, 1)

    def forward(self, hidden_states: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        hidden_states = self.out_proj(hidden_states)
        hidden_states = self.avg_pool(hidden_states, masks)
        hidden_states = torch.sigmoid(hidden_states)
        return hidden_states

    def avg_pool(self, hidden_states, mask):
        length = torch.sum(mask, 1, keepdim=True).float()
        mask = mask.unsqueeze(2)
        hidden = hidden_states.masked_fill(mask == 0, 0.0)
        avg_hidden = torch.sum(hidden, 1) / length

        return avg_hidden
    
class CLBartForConditionalGeneration(BartForConditionalGeneration):
    base_model_prefix = "model"
    _keys_to_ignore_on_load_missing = [
        r"final_logits_bias",
        r"lm_head.weight",
        "encoder.embed_tokens.weight",
        "decoder.embed_tokens.weight",
    ]


    def __init__(self, config: BartConfig):
        super().__init__(config)
        self.model = BartModel(config)
        self.tokenizer = BartTokenizer.from_pretrained("cogint/in-boxbart")
        self.contrastive_head = ContrastiveHead(config.d_model, config.dropout)

        # Initialize weights and apply final processing
        self.post_init()
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,

        neg_ids: Optional[torch.LongTensor] = None,
        neg_num_total: Optional[int]=1,

        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, Seq2SeqLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            if use_cache:
                logger.warning("The `use_cache` argument is changed to `False` since `labels` is provided.")
            use_cache = False
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        lm_logits = self.lm_head(outputs[0])
        lm_logits = lm_logits + self.final_logits_bias.to(lm_logits.device)

        masked_lm_loss = None
        contrastive_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))

            contrastive_loss = self.contrastive(
                outputs=outputs,
                labels=labels,
                attention_mask=attention_mask,
                neg_ids=neg_ids,
                expand_size=neg_num_total,

                decoder_attention_mask=decoder_attention_mask,
                decoder_head_mask=decoder_head_mask,
                cross_attn_head_mask=cross_attn_head_mask,
                past_key_values=past_key_values,
                decoder_inputs_embeds=decoder_inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                loss_fct=loss_fct,
            )

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss,contrastive_loss,) + output) if masked_lm_loss is not None else output
        if contrastive_loss is None:
            return ContrastiveSeq2SeqLMOutput(
                loss=masked_lm_loss,
                mlm_loss=masked_lm_loss,
                cl_loss=contrastive_loss,
                logits=lm_logits,
                past_key_values=outputs.past_key_values,
                decoder_hidden_states=outputs.decoder_hidden_states,
                decoder_attentions=outputs.decoder_attentions,
                cross_attentions=outputs.cross_attentions,
                encoder_last_hidden_state=outputs.encoder_last_hidden_state,
                encoder_hidden_states=outputs.encoder_hidden_states,
                encoder_attentions=outputs.encoder_attentions,
            )
        else:
            return ContrastiveSeq2SeqLMOutput(
                loss=masked_lm_loss+contrastive_loss,
                mlm_loss=masked_lm_loss,
                cl_loss=contrastive_loss,
                logits=lm_logits,
                past_key_values=outputs.past_key_values,
                decoder_hidden_states=outputs.decoder_hidden_states,
                decoder_attentions=outputs.decoder_attentions,
                cross_attentions=outputs.cross_attentions,
                encoder_last_hidden_state=outputs.encoder_last_hidden_state,
                encoder_hidden_states=outputs.encoder_hidden_states,
                encoder_attentions=outputs.encoder_attentions,
            )
    
    def contrastive(
        self,
        outputs,
        labels,
        attention_mask,
        neg_ids,
        expand_size,

        decoder_attention_mask,
        decoder_head_mask,
        cross_attn_head_mask,
        past_key_values,
        decoder_inputs_embeds,
        use_cache,
        output_attentions,
        output_hidden_states,
        return_dict,
        loss_fct
    ):
        pos_label_mask = labels != self.tokenizer.pad_token_id
        pos_emb = self.contrastive_head(outputs.last_hidden_state, pos_label_mask)

        decoder_input_ids = shift_tokens_right(
            neg_ids, self.config.pad_token_id, self.config.decoder_start_token_id
        )

        bs = labels.size(0)

        expanded_return_idx = (
            torch.arange(attention_mask.shape[0]).view(-1, 1).repeat(1, expand_size).view(-1).to(attention_mask.device)
        )
        encoder_outputs = NewBaseModelOutput(
            last_hidden_state = outputs.encoder_last_hidden_state,
            hidden_states=outputs.encoder_hidden_states,
            attentions=outputs.encoder_attentions,
        )

        encoder_outputs["last_hidden_state"] = encoder_outputs.last_hidden_state.index_select(
            0, expanded_return_idx.to(encoder_outputs.last_hidden_state.device)
        )

        attention_mask = attention_mask.index_select(0, expanded_return_idx).to(attention_mask.device)
 
        
        decoder = self.get_decoder()

        decoder_outputs = decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=attention_mask,

            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        neg_label_mask = neg_ids != self.tokenizer.pad_token_id
        neg_emb = self.contrastive_head(decoder_outputs.last_hidden_state, neg_label_mask).view(bs, expand_size)


        all_logit = torch.cat([pos_emb,neg_emb], dim=1)
        l = torch.zeros([bs], dtype=torch.long, device=neg_emb.device)
        cl_loss = loss_fct(all_logit, l)
        return cl_loss