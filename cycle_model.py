from torch import nn
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch
from torch.nn.functional import gumbel_softmax
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers.modeling_outputs import (
    ModelOutput, 
    BaseModelOutputWithPastAndCrossAttentions, 
    Seq2SeqModelOutput, 
    BaseModelOutput, 
    Seq2SeqLMOutput,
)
from transformers import BartConfig, BartForConditionalGeneration
from model import CLBartForConditionalGeneration


 
    
class ContrastiveHeadAvg(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(
        self,
        inner_dim: int,
        pooler_dropout: float,
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(inner_dim, inner_dim)

    def forward(self, hidden_states: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        hidden_states = self.out_proj(hidden_states)
        hidden_states = self.avg_pool(hidden_states, masks)
        hidden_states = torch.sigmoid(hidden_states)
        return hidden_states

    def avg_pool(self, hidden_states, mask):
        if mask is None:
            length = hidden_states.size(1)
            avg_hidden = torch.sum(hidden_states, 1) / length
        else:
            length = torch.sum(mask, 1, keepdim=True).float()
            mask = mask.unsqueeze(2)
            hidden = hidden_states.masked_fill(mask == 0, 0.0)
            avg_hidden = torch.sum(hidden, 1) / length

        return avg_hidden

    
class CycleV1_cl(nn.Module):
    def __init__(self, config: BartConfig, device):
        super(CycleV1_cl, self).__init__()

        self.contrastive_head = ContrastiveHeadAvg(config.d_model, config.dropout)
        self.model1 = CLBartForConditionalGeneration.from_pretrained(
            "cogint/in-boxbart", 
            config=config)
        self.model2 = BartForConditionalGeneration.from_pretrained(
            "cogint/in-boxbart", 
            config=config)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.cross_entropy = nn.CrossEntropyLoss()
        self.device = device
    

    def forward(
            self,
            batch
            ):
        device = self.device
        input_ids = batch['input_ids'].to(device)
        attention_masks = batch['attention_masks'].to(device)
        target_ids = batch['target_ids'].to(device)

        input_ids_ = batch['input_ids_i'].to(device)

        neg_ids = batch["neg_ids"].to(self.device)
        neg_num_total = batch["neg_num_total"]

        results_f = self.model1(
                input_ids=input_ids,
                attention_mask=attention_masks,


                neg_ids=neg_ids,
                neg_num_total=neg_num_total,

                labels=target_ids,
                return_dict=True
            )
        cl_loss_decoder = results_f['cl_loss']
        mlm_loss = results_f['mlm_loss']

        logits_f = results_f['logits']
        soft_lf = gumbel_softmax(logits_f,dim=-1)
        lf_embed = soft_lf @self.model2.model.shared.weight.clone()

        results_fb = self.model2(
                inputs_embeds=lf_embed,
                labels=input_ids_,
                return_dict=True
            )
        rec_loss_f = results_fb['loss']
        result = { 
            'loss': mlm_loss * 5 + cl_loss_decoder * 0.2 + rec_loss_f,
            'rec_loss_f': rec_loss_f,
            'sup_loss_f': mlm_loss,
            'cl_loss_decoder': cl_loss_decoder,
        }
        return result
    