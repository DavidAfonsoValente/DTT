import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from typing import Optional, Tuple
from dataclasses import dataclass

@dataclass
class CausalLMOutputWithGates(CausalLMOutputWithCrossAttentions):
    gates: Optional[torch.FloatTensor] = None

class DTTModel(GPT2LMHeadModel):
    def __init__(self, config):
        super().__init__(config)
        self.gate_network = nn.Linear(config.n_embd, 1)
        nn.init.xavier_normal_(self.gate_network.weight)
        self.gate_network.bias.data.fill_(-5.0)  # Initial token-dominant (low g)

        self.temperature = 2.0

        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.add_special_tokens({
            'additional_special_tokens': ['[bot]', '[eot]'],
            'pad_token': '<pad>'
        })
        self.resize_token_embeddings(len(self.tokenizer))

        with torch.no_grad():
            mean_emb = self.transformer.wte.weight[:-2].mean(dim=0)
            noise = torch.randn_like(mean_emb) * 0.02
            self.transformer.wte.weight[-2] = mean_emb + noise
            self.transformer.wte.weight[-1] = mean_emb + noise.clone()

        self.bot_id = self.tokenizer.convert_tokens_to_ids('[bot]')
        self.eot_id = self.tokenizer.convert_tokens_to_ids('[eot]')
        self.pad_id = self.tokenizer.pad_token_id
        self.debug = False

    def set_temperature(self, tau):
        self.temperature = max(0.1, tau)

    def gumbel_sigmoid(self, logit, temperature, training):
        if training:
            u = torch.rand_like(logit)
            gumbel_noise = -torch.log(-torch.log(u + 1e-20) + 1e-20)
            return F.sigmoid((logit + gumbel_noise) / temperature)
        else:
            return F.sigmoid(logit)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_hidden_states: Optional[bool] = None,
        **kwargs
    ) -> CausalLMOutputWithGates:
        batch_size, seq_len = input_ids.shape
        position_ids = torch.arange(0, seq_len, dtype=torch.long, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        gates = []
        logits = []
        past_key_values = None
        h_prev = None
        g_prev = torch.zeros(batch_size, device=input_ids.device)
        extended_attention_mask = self.transformer.get_extended_attention_mask(attention_mask, input_ids.shape) if attention_mask is not None else None

        # Collect last_hidden_state and optionally all hidden_states
        hidden_list = []  # For last_hidden_state (all tokens)
        all_hidden_states = [] if output_hidden_states else None
        if output_hidden_states:
            all_hidden_states = [[] for _ in range(self.config.num_hidden_layers + 1)]  # +1 for input embeds

        for t in range(seq_len):
            current_input_id = input_ids[:, t:t+1]
            current_position_id = position_ids[:, t:t+1]
            current_mask = extended_attention_mask[:, :, t:t+1, :t+1] if extended_attention_mask is not None else None

            e = self.transformer.wte(current_input_id)
            if t > 0:
                sqrt_g = torch.sqrt(g_prev).unsqueeze(-1)
                sqrt_1_g = torch.sqrt(1 - g_prev).unsqueeze(-1)
                e = sqrt_g * h_prev.unsqueeze(1) + sqrt_1_g * e

            transformer_outputs = self.transformer(
                inputs_embeds=e,
                position_ids=current_position_id,
                attention_mask=current_mask,
                past_key_values=past_key_values,
                use_cache=True,
                output_hidden_states=output_hidden_states,
                **kwargs
            )
            past_key_values = transformer_outputs.past_key_values
            hidden_state = transformer_outputs.last_hidden_state[:, -1, :]
            hidden_list.append(hidden_state.unsqueeze(1))  # Collect for potential full last_hidden_state

            if output_hidden_states:
                # transformer_outputs.hidden_states is tuple: (input_embeds, layer1, ..., layerN)
                for layer_idx, layer_hidden in enumerate(transformer_outputs.hidden_states):
                    all_hidden_states[layer_idx].append(layer_hidden[:, -1:, :])  # Append current token's state

            gate_logit = self.gate_network(hidden_state).squeeze(-1)
            g_prev = self.gumbel_sigmoid(gate_logit, self.temperature, self.training)
            gates.append(g_prev.unsqueeze(1))
            logit = self.lm_head(hidden_state)
            logits.append(logit)

            h_prev = hidden_state

        logits = torch.cat(logits, dim=1)
        gates = torch.cat(gates, dim=1)

        # Concatenate collected hidden states
        last_hidden_state = torch.cat(hidden_list, dim=1)
        hidden_states = None
        if output_hidden_states:
            hidden_states = tuple(torch.cat(layer_list, dim=1) for layer_list in all_hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :]
            shift_labels = labels[:, 1:]
            loss = F.cross_entropy(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))

        return CausalLMOutputWithGates(
            loss=loss,
            logits=logits,
            past_key_values=past_key_values,
            hidden_states=hidden_states,
            last_hidden_state=last_hidden_state,
            gates=gates,
        )

    def generate(self, input_ids, max_length=256, do_sample=True, temperature=0.8, top_p=0.95, return_gates=False, return_logprobs=False, training=False):
        if self.debug:
            print(f"[DEBUG] Starting generation with input_ids shape: {input_ids.shape}, do_sample={do_sample}, training={training}")
        input_ids = input_ids.to(self.device)
        batch_size = input_ids.size(0)
        generated_ids = input_ids.clone()
        gates_list = []
        logprobs_list = [] if return_logprobs else None
        past_key_values = None
        h_prev = None
        g_prev = torch.zeros(batch_size, device=self.device)
        position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long, device=self.device).unsqueeze(0).expand(batch_size, -1)
        attention_mask = torch.ones(batch_size, input_ids.size(1), dtype=torch.long, device=self.device)

        for step in range(max_length - input_ids.size(1)):
            if step == 0:
                outputs = self(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
                hidden = outputs.hidden_states[-1][:, -1, :]  # Last layer, last token
                gate_logit = self.gate_network(hidden).squeeze(-1)
                g_prev = self.gumbel_sigmoid(gate_logit, self.temperature, training=training)
                gates_list.append(g_prev.unsqueeze(1))
                next_token_logits = outputs.logits[:, -1, :] / temperature
                past_key_values = outputs.past_key_values
            else:
                e = torch.sqrt(1 - g_prev).unsqueeze(-1) * self.transformer.wte(next_token.unsqueeze(1)) + torch.sqrt(g_prev).unsqueeze(-1) * h_prev.unsqueeze(1)
                current_position_id = (position_ids[:, -1] + 1).unsqueeze(1)
                current_mask = torch.ones(batch_size, 1, dtype=torch.long, device=self.device)
                transformer_outputs = self.transformer(inputs_embeds=e, position_ids=current_position_id, attention_mask=current_mask, past_key_values=past_key_values, use_cache=True)
                past_key_values = transformer_outputs.past_key_values
                hidden = transformer_outputs.last_hidden_state[:, -1, :]
                gate_logit = self.gate_network(hidden).squeeze(-1)
                g_prev = self.gumbel_sigmoid(gate_logit, self.temperature, training=training)
                gates_list.append(g_prev.unsqueeze(1))
                next_token_logits = self.lm_head(hidden) / temperature

            next_token_logits = torch.clamp(next_token_logits, -1e4, 1e4)
            if do_sample:
                filtered_logits = next_token_logits.clone()
                sorted_logits, sorted_indices = torch.sort(filtered_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                filtered_logits[indices_to_remove] = float('-inf')
                probs = F.softmax(filtered_logits, dim=-1)
                probs = torch.nan_to_num(probs, nan=1.0 / probs.size(-1))
                next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1)

            if return_logprobs:
                log_softmax = F.log_softmax(filtered_logits if do_sample else next_token_logits, dim=-1)
                logp = log_softmax[torch.arange(batch_size), next_token]
                logprobs_list.append(logp)

            generated_ids = torch.cat([generated_ids, next_token.unsqueeze(1)], dim=1)
            position_ids = torch.cat([position_ids, (position_ids[:, -1] + 1).unsqueeze(1)], dim=1)
            attention_mask = torch.cat([attention_mask, torch.ones(batch_size, 1, dtype=torch.long, device=self.device)], dim=1)
            h_prev = hidden

            if step > 10 and torch.all(generated_ids[:, -10:] == generated_ids[:, -1]):
                break

        if self.debug:
            print(f"[DEBUG] Generation completed with {generated_ids.size(1) - input_ids.size(1)} steps")

        returns = [generated_ids]
        if return_gates:
            gates_cat = torch.cat(gates_list, dim=1) if gates_list else torch.empty(batch_size, 0, device=self.device)
            returns.append(gates_cat)
        if return_logprobs:
            logprobs_cat = torch.stack(logprobs_list, dim=1) if logprobs_list else torch.empty(batch_size, 0, device=self.device)
            returns.append(logprobs_cat)
        return returns if len(returns) > 1 else returns[0]