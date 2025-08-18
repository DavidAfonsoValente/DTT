import torch
import torch.nn as nn
from torch.nn.functional import sigmoid, cross_entropy, softmax
from transformers import GPT2LMHeadModel, GPT2Tokenizer

class DTTModel(GPT2LMHeadModel):
    def __init__(self, config):
        super().__init__(config)
        self.gate_weight = nn.Parameter(torch.randn(1, config.n_embd))  # W
        self.gate_bias = nn.Parameter(torch.tensor(0.0))  # b
        self.special_tokens = {'bot': '[bot]', 'eot': '[eot]'}
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.add_tokens(list(self.special_tokens.values()))
        self.resize_token_embeddings(len(self.tokenizer))
        self.bot_id = self.tokenizer.convert_tokens_to_ids(self.special_tokens['bot'])
        self.eot_id = self.tokenizer.convert_tokens_to_ids(self.special_tokens['eot'])
        self.dummy_id = self.tokenizer.pad_token_id
        self.temperature = 2.0

        def attn_hook(module, args):
            q, k, v = args
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (k.size(-1)**0.5)
            if hasattr(self, 'noisy_mask') and self.noisy_mask is not None:
                seq_len = attn_scores.size(-1)
                noisy_mask_exp = self.noisy_mask.unsqueeze(1).unsqueeze(2).expand(-1, attn_scores.size(1), seq_len, seq_len)
                attn_scores = torch.where(noisy_mask_exp & noisy_mask_exp.transpose(-2, -1), attn_scores * 0.3, attn_scores)
            attn_probs = softmax(attn_scores, dim=-1)
            return torch.matmul(attn_probs, v)

        for layer in self.transformer.h:
            layer.attn.register_forward_hook(attn_hook)

    def gumbel_sigmoid(self, logit, temperature, training):
        if training:
            u = torch.rand_like(logit)
            gumbel_noise = -torch.log(-torch.log(u + 1e-20) + 1e-20)
            return sigmoid((logit + gumbel_noise) / temperature)
        return sigmoid(logit)

    def forward(self, input_ids=None, attention_mask=None, labels=None, input_embeds=None, noisy_mask=None, **kwargs):
        if noisy_mask is not None:
            self.noisy_mask = noisy_mask.float()

        if input_embeds is not None:
            outputs = super().forward(inputs_embeds=input_embeds, attention_mask=attention_mask, output_hidden_states=True, **kwargs)
        else:
            outputs = super().forward(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, **kwargs)

        hidden_states = outputs.hidden_states[-1]
        gate_logits = torch.matmul(hidden_states, self.gate_weight.t()) + self.gate_bias
        gate_logits = gate_logits.squeeze(-1)
        gates = self.gumbel_sigmoid(gate_logits, self.temperature, self.training)

        if labels is not None:
            shift_logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            outputs['loss'] = loss

        outputs['gates'] = gates
        self.noisy_mask = None
        return outputs

    def generate(self, input_ids, max_length=512, do_sample=True, top_p=0.95, temperature=1.0, return_gates=False, **kwargs):
        batch_size = input_ids.size(0)
        input_embeds = self.transformer.wte(input_ids)
        generated_ids = input_ids.clone()
        gates_list = []

        for _ in range(max_length - input_ids.size(1)):
            outputs = self.forward(inputs_embeds=input_embeds)
            hidden = outputs['hidden_states'][-1][:, -1, :]
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