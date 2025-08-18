import torch
import torch.nn as nn
from torch.nn.functional import sigmoid, relu, cross_entropy, kl_div, softmax
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
from transformers.modeling_utils import ModuleUtilsMixin
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention
import weakref
from typing import Optional, Tuple
from dataclasses import dataclass
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

@dataclass
class CausalLMOutputWithGates(CausalLMOutputWithCrossAttentions):
    gates: Optional[torch.FloatTensor] = None

class CustomGPT2Attention(GPT2Attention):
    def __init__(self, config, is_cross_attention=False, layer_idx=None):
        super().__init__(config, is_cross_attention, layer_idx)
        self.model = None  # Set later

    def forward(self, hidden_states, layer_past=None, attention_mask=None, head_mask=None, encoder_hidden_states=None, encoder_attention_mask=None, use_cache=False, output_attentions=False):
        query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        if layer_past is not None:
            past_key, past_value = layer_past
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)

        if use_cache is True:
            present = (key, value)
        else:
            present = None

        attn_scores = torch.matmul(query, key.transpose(-1, -2)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))

        # Apply noisy scaling if noisy_mask is set
        if self.model is not None:
            model = self.model()
            if model is not None and hasattr(model, 'noisy_mask') and model.noisy_mask is not None:
                seq_len = attn_scores.size(-1)
                noisy_mask_exp = model.noisy_mask.unsqueeze(1).unsqueeze(2).expand(-1, self.num_heads, seq_len, seq_len)
                attn_scores = torch.where(noisy_mask_exp & noisy_mask_exp.transpose(-2, -1), attn_scores * 0.3, attn_scores)

        attn_scores = attn_scores + attention_mask
        attn_probs = torch.nn.functional.softmax(attn_scores, dim=-1)
        attn_probs = self.attn_dropout(attn_probs)

        if head_mask is not None:
            attn_probs = attn_probs * head_mask

        attn_output = torch.matmul(attn_probs, value)
        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_scores,)

        return outputs

class DTTModel(GPT2LMHeadModel):
    def __init__(self, config):
        super().__init__(config)
        self.gate_weight = nn.Parameter(torch.randn(1, config.n_embd))  # W
        self.gate_bias = nn.Parameter(torch.tensor(0.0))  # b
        self.special_tokens = {'bot': '[bot]', 'eot': '[eot]'}
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.add_special_tokens({
            'additional_special_tokens': list(self.special_tokens.values()),
            'pad_token': '<pad>'
        })
        self.resize_token_embeddings(len(self.tokenizer))
        self.bot_id = self.tokenizer.convert_tokens_to_ids(self.special_tokens['bot'])
        self.eot_id = self.tokenizer.convert_tokens_to_ids(self.special_tokens['eot'])
        self.dummy_id = self.tokenizer.pad_token_id
        self.temperature = 2.0

        for i, layer in enumerate(self.transformer.h):
            custom_attn = CustomGPT2Attention(self.config, is_cross_attention=False, layer_idx=i)
            custom_attn.model = weakref.ref(self)
            layer.attn = custom_attn

    def gumbel_sigmoid(self, logit, temperature, training):
        if training:
            u = torch.rand_like(logit)
            gumbel_noise = -torch.log(-torch.log(u + 1e-20) + 1e-20)
            return sigmoid((logit + gumbel_noise) / temperature)
        return sigmoid(logit)

    def forward(self, input_ids=None, attention_mask=None, labels=None, input_embeds=None, noisy_mask=None, **kwargs):
        if noisy_mask is not None:
            self.noisy_mask = noisy_mask  # Keep as bool, remove .float()

        if input_embeds is not None:
            transformer_outputs = super().forward(inputs_embeds=input_embeds, attention_mask=attention_mask, output_hidden_states=True, **kwargs)
        else:
            transformer_outputs = super().forward(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, **kwargs)

        hidden_states = transformer_outputs.hidden_states[-1]
        gate_logits = torch.matmul(hidden_states, self.gate_weight.t()) + self.gate_bias
        gate_logits = gate_logits.squeeze(-1)
        gates = self.gumbel_sigmoid(gate_logits, self.temperature, self.training)

        loss = None
        if labels is not None:
            shift_logits = transformer_outputs.logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        self.noisy_mask = None
        return CausalLMOutputWithGates(
            loss=loss,
            logits=transformer_outputs.logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
            gates=gates,
        )

    def generate(self, input_ids, max_length=512, do_sample=True, top_p=0.95, temperature=1.0, return_gates=False, **kwargs):
        batch_size = input_ids.size(0)
        input_embeds = self.transformer.wte(input_ids)
        generated_ids = input_ids.clone()
        gates_list = []

        for _ in range(max_length - input_ids.size(1)):
            outputs = self.forward(inputs_embeds=input_embeds)
            hidden = outputs.hidden_states[-1][:, -1, :]
            gate_logit = torch.matmul(hidden, self.gate_weight.t()) + self.gate_bias
            gate = self.gumbel_sigmoid(gate_logit.squeeze(-1), self.temperature, self.training)
            gates_list.append(gate.unsqueeze(1))

            next_token_logits = outputs.logits[:, -1, :] / temperature
            if do_sample:
                filtered_logits = next_token_logits.clone()
                sorted_logits, sorted_indices = torch.sort(filtered_logits, descending=True)
                cumulative_probs = torch.cumsum(softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                filtered_logits[indices_to_remove] = float('-inf')
                next_token = torch.multinomial(softmax(filtered_logits, dim=-1), num_samples=1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            if gate.mean() > 0.5:
                next_embed = gate.unsqueeze(-1) * hidden + (1 - gate).unsqueeze(-1) * self.transformer.wte(torch.full((batch_size, 1), self.dummy_id, device=hidden.device))
                generated_ids = torch.cat([generated_ids, torch.full((batch_size, 1), self.dummy_id, device=generated_ids.device)], dim=1)
            else:
                next_embed = self.transformer.wte(next_token)
                generated_ids = torch.cat([generated_ids, next_token], dim=1)

            input_embeds = torch.cat([input_embeds, next_embed], dim=1)

        if return_gates:
            return generated_ids, torch.cat(gates_list, dim=1)
        return generated_ids

    def set_temperature(self, tau):
        self.temperature = max(0.1, tau)