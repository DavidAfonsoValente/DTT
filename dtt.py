"""
Dynamic Thinking Tokens (DTT) Framework - dtt.py

This module extends Coconut's latent reasoning mechanism with GRPO reinforcement learning.
"""

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import torch.distributed as dist
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any, Union

@dataclass
class Outputs:
    loss: Optional[torch.Tensor] = None
    inputs_embeds: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None
    rewards: Optional[torch.Tensor] = None
    advantages: Optional[torch.Tensor] = None
    hidden_states_list: Optional[List[torch.Tensor]] = None

class DTT(nn.Module):
    """
    Dynamic Thinking Tokens (DTT) Framework
    
    Extends Coconut's latent reasoning mechanism with GRPO reinforcement learning.
    """
    
    def __init__(
        self, 
        base_causallm, 
        latent_id, 
        start_id, 
        end_id, 
        eos_token_id,
        max_latent_steps=20,
        phase="final",
    ):
        super().__init__()
        self.base_causallm = base_causallm
        self.latent_id = latent_id
        self.start_id = start_id
        self.end_id = end_id
        self.eos_token_id = eos_token_id
        self.max_latent_steps = max_latent_steps
        self.phase = phase
        
        # Get the embedding layer from the base model
        self.embedding = self.base_causallm.get_input_embeddings()
        
        # Counter for generation forward passes (used for synced GPUs)
        self.gen_forward_cnt = 0
        
        # Initialize reward weights based on phase
        self._set_phase_weights(phase)
    
    def _set_phase_weights(self, phase):
        """Set reward weights based on training phase"""
        if phase == "warmup":
            self.w_binary = 0.7
            self.w_crs = 0.2
            self.w_lcr = 0.1
            self.w_ede = 0.0
            self.max_latent_steps = 50
        elif phase == "core":
            self.w_binary = 0.8
            self.w_crs = 0.15
            self.w_lcr = 0.0
            self.w_ede = 0.05
            # max_latent_steps is annealed from 50 to 20 during this phase
        else:  # final
            self.w_binary = 1.0
            self.w_crs = 0.0
            self.w_lcr = 0.0
            self.w_ede = 0.0
            self.max_latent_steps = 20
    
    def set_phase(self, phase, current_step=None, total_steps=None):
        """Update phase and associated parameters"""
        self.phase = phase
        self._set_phase_weights(phase)
        
        # Anneal max_latent_steps during core phase
        if phase == "core" and current_step is not None and total_steps is not None:
            progress = current_step / total_steps
            self.max_latent_steps = int(50 - (30 * progress))
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        position_ids=None,
        past_key_values=None,
        output_hidden_states=True,
        return_dict=True,
        compute_rewards=False,
    ):
        """
        Forward pass with latent reasoning and optional reward computation
        """
        batch_size, seq_len = input_ids.shape
        
        # Get embeddings for input tokens
        inputs_embeds = self.embedding(input_ids)
        
        # Find positions of latent tokens (for compatibility, though not used in RL generation)
        latent_lists = []
        for i in range(batch_size):
            latent_mask = (input_ids[i] == self.latent_id).nonzero().squeeze(-1).tolist()
            if not isinstance(latent_mask, list):
                latent_mask = [latent_mask]
            latent_lists.append(latent_mask)
        
        max_n_latents = max([len(l) for l in latent_lists]) if latent_lists else 0
        max_n_latents = min(max_n_latents, self.max_latent_steps)
        
        # Initialize storage for logits and hidden states
        logits = []
        hidden_states_list = []
        
        # Process tokens up to the first latent token
        next_compute_range = (0, seq_len)
        for i in range(batch_size):
            if latent_lists[i]:
                next_compute_range = (0, min(next_compute_range[1], latent_lists[i][0] + 1))
        
        # Initial forward pass
        outputs = self.base_causallm(
            inputs_embeds=inputs_embeds[:, next_compute_range[0]:next_compute_range[1], :],
            attention_mask=attention_mask[:, :next_compute_range[1]],
            position_ids=position_ids[:, next_compute_range[0]:next_compute_range[1]] if position_ids is not None else None,
            past_key_values=past_key_values,
            output_hidden_states=True,
        )
        
        logits.append(outputs.logits)
        hidden_states = outputs.hidden_states[-1]
        hidden_states_list.append(hidden_states)
        
        # Process latent tokens (for supervised compatibility)
        for pass_idx in range(max_n_latents):
            next_compute_range = (next_compute_range[1], seq_len)
            for i in range(batch_size):
                if len(latent_lists[i]) > pass_idx + 1:
                    next_compute_range = (
                        next_compute_range[0],
                        min(next_compute_range[1], latent_lists[i][pass_idx + 1] + 1),
                    )
            
            if next_compute_range[0] == next_compute_range[1]:
                break
            
            filling_indices = [
                (instance_idx, mask_list[pass_idx])
                for instance_idx, mask_list in enumerate(latent_lists)
                if len(mask_list) > pass_idx
            ]
            
            tensor_list = [
                [
                    inputs_embeds[batch_idx, pos, :]
                    for pos in range(inputs_embeds.shape[1])
                ]
                for batch_idx in range(inputs_embeds.shape[0])
            ]
            
            for idx_pair in filling_indices:
                batch_idx, token_idx = idx_pair
                tensor_list[batch_idx][token_idx] = hidden_states[batch_idx, -1, :]
            
            inputs_embeds = torch.stack(
                [
                    torch.stack(tensor_list[batch_idx])
                    for batch_idx in range(inputs_embeds.shape[0])
                ]
            )
            
            outputs = self.base_causallm(
                inputs_embeds=inputs_embeds[:, next_compute_range[0]:next_compute_range[1], :],
                attention_mask=attention_mask[:, :next_compute_range[1]],
                position_ids=position_ids[:, next_compute_range[0]:next_compute_range[1]] if position_ids is not None else None,
                past_key_values=None,
                output_hidden_states=True,
            )
            
            logits.append(outputs.logits)
            hidden_states = outputs.hidden_states[-1]
            hidden_states_list.append(hidden_states)
        
        # Final pass for any remaining tokens
        if next_compute_range[1] < seq_len:
            outputs = self.base_causallm(
                inputs_embeds=inputs_embeds[:, next_compute_range[1]:, :],
                attention_mask=attention_mask[:, next_compute_range[1]:],
                position_ids=position_ids[:, next_compute_range[1]:] if position_ids is not None else None,
                past_key_values=None,
                output_hidden_states=True,
            )
            
            logits.append(outputs.logits)
            hidden_states_list.append(outputs.hidden_states[-1])
        
        self.gen_forward_cnt += max_n_latents + 1
        
        # Concatenate logits
        logits = torch.cat(logits, dim=1)
        
        # Compute loss if labels are provided (for supervised compatibility)
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )
        
        # Compute rewards if requested
        rewards = None
        advantages = None
        if compute_rewards and len(hidden_states_list) > 1:
            rewards = self._compute_rewards(hidden_states_list, logits, labels)
            
            if rewards is not None and rewards.shape[0] > 1:
                advantages = self._compute_advantages(rewards)
        
        return Outputs(
            loss=loss,
            inputs_embeds=inputs_embeds,
            logits=logits,
            rewards=rewards,
            advantages=advantages,
            hidden_states_list=hidden_states_list
        )
    
    def _compute_rewards(self, hidden_states_list, logits, labels):
        """
        Compute rewards based on current phase
        """
        batch_size = hidden_states_list[0].shape[0]
        rewards = torch.zeros(batch_size, device=hidden_states_list[0].device)
        
        # Binary reward component (placeholder, computed externally in RL)
        binary_rewards = torch.rand(batch_size, device=rewards.device)
        
        # Latent Consistency Regularization (LCR)
        lcr_rewards = torch.zeros_like(rewards)
        if self.w_lcr > 0 and len(hidden_states_list) > 1:
            for i in range(len(hidden_states_list) - 1):
                h1 = hidden_states_list[i]
                h2 = hidden_states_list[i + 1]
                h1_norm = h1 / (h1.norm(dim=-1, keepdim=True) + 1e-8)
                h2_norm = h2 / (h2.norm(dim=-1, keepdim=True) + 1e-8)
                cos_sim = (h1_norm * h2_norm).sum(dim=-1).mean(dim=-1)
                lcr_rewards += cos_sim
            lcr_rewards = lcr_rewards / (len(hidden_states_list) - 1)
        
        # Contrastive Reward Shaping (CRS)
        crs_rewards = torch.zeros_like(rewards)
        if self.w_crs > 0 and batch_size > 1:
            crs_rewards = torch.softmax(binary_rewards, dim=0) - 0.5
        
        # Entropy-Driven Exploration (EDE)
        ede_rewards = torch.zeros_like(rewards)
        if self.w_ede > 0:
            ede_rewards = torch.rand_like(rewards) * 0.1
        
        # Efficiency reward
        n_latent_steps = len(hidden_states_list) - 1
        efficiency_reward = 0.98 ** n_latent_steps
        
        rewards = (
            self.w_binary * binary_rewards +
            self.w_crs * crs_rewards +
            self.w_lcr * lcr_rewards +
            self.w_ede * ede_rewards +
            0.1 * efficiency_reward
        )
        
        return rewards
    
    def _compute_advantages(self, rewards):
        """Compute advantages for GRPO"""
        mean_reward = rewards.mean()
        std_reward = rewards.std() + 1e-8
        advantages = (rewards - mean_reward) / std_reward
        return advantages
    
    def generate(
        self,
        input_ids,
        attention_mask,
        max_new_tokens=16,
        max_latent_steps=20,  # Renamed to max_latent_steps for clarity
        synced_gpus=False,
        **kwargs
    ):
        """
        Generate text with dynamic latent reasoning for batch sizes > 1.
        Each sequence can have a different number of latent steps, learned by producing <|end-latent|>.
        """
        batch_size = input_ids.shape[0]
        device = input_ids.device
        tokens = input_ids.clone()
        current_attention_mask = attention_mask.clone()

        # Initial forward pass up to <|start-latent|>
        outputs = self.base_causallm(
            input_ids=tokens,
            attention_mask=current_attention_mask,
            use_cache=True
        )
        past_key_values = outputs.past_key_values
        current_hidden = outputs.hidden_states[-1][:, -1, :]  # [batch_size, hidden_dim]

        # Latent steps with dynamic stopping
        active = torch.ones(batch_size, dtype=torch.bool, device=device)  # Mask for active latent sequences
        step = 0
        while step < max_latent_steps and active.any():
            # Process only active sequences
            inputs_embeds = current_hidden[active].unsqueeze(1)  # [active_batch, 1, hidden_dim]
            active_past = [pv[active] for pv in past_key_values]
            outputs = self.base_causallm(
                inputs_embeds=inputs_embeds,
                past_key_values=active_past,
                use_cache=True
            )
            # Update past_key_values for active sequences
            for i, pv in enumerate(outputs.past_key_values):
                past_key_values[i][active] = pv
            current_hidden_active = outputs.hidden_states[-1][:, -1, :]  # [active_batch, hidden_dim]

            # Compute logits and determine if <|end-latent|> is predicted (greedy decoding)
            logits = self.base_causallm.lm_head(current_hidden_active)
            virtual_tokens = torch.argmax(logits, dim=-1)  # Greedy selection

            # Update active mask: deactivate sequences predicting <|end-latent|>
            end_predicted = virtual_tokens == self.end_id
            active[active.clone()] = ~end_predicted

            # Update hidden states for still-active sequences
            if (~end_predicted).any():
                current_hidden[active] = current_hidden_active[~end_predicted]

            step += 1

        # Generate <|end-latent|> token for all sequences
        end_token = torch.full((batch_size, 1), self.end_id, device=device)
        tokens = torch.cat([tokens, end_token], dim=1)
        current_attention_mask = torch.cat(
            [current_attention_mask, torch.ones((batch_size, 1), device=device)],
            dim=1
        )

        # Generate answer tokens
        for _ in range(max_new_tokens):
            outputs = self.base_causallm(
                input_ids=tokens[:, -1:],
                attention_mask=current_attention_mask,
                past_key_values=past_key_values,
                use_cache=True
            )
            next_token_logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
            tokens = torch.cat([tokens, next_token], dim=1)
            current_attention_mask = torch.cat(
                [current_attention_mask, torch.ones((batch_size, 1), device=device)],
                dim=1
            )
            past_key_values = outputs.past_key_values
            if (next_token == self.eos_token_id).all():
                break

        if synced_gpus:
            additional_passes = max_new_tokens + max_latent_steps - self.gen_forward_cnt
            while self.gen_forward_cnt < max_new_tokens + max_latent_steps:
                self.gen_forward_cnt += 1
                _ = self.base_causallm(input_ids=tokens[:, -1:], attention_mask=current_attention_mask)

        return tokens

    def generate_with_log_probs(
        self,
        input_ids,
        attention_mask,
        max_new_tokens=16,
        max_latent_steps=20,  # Renamed to max_latent_steps for clarity
        temperature=1.0,
        synced_gpus=False,
        **kwargs
    ):
        """
        Generate text with log probabilities for RL training, supporting dynamic latent steps.
        Each sequence can have a different number of latent steps, learned by producing <|end-latent|>.
        """
        batch_size = input_ids.shape[0]
        device = input_ids.device
        tokens = input_ids.clone()
        current_attention_mask = attention_mask.clone()
        log_probs = []

        # Initial forward pass
        outputs = self.base_causallm(
            input_ids=tokens,
            attention_mask=current_attention_mask,
            use_cache=True
        )
        past_key_values = outputs.past_key_values
        current_hidden = outputs.hidden_states[-1][:, -1, :]

        # Latent steps with dynamic stopping
        active = torch.ones(batch_size, dtype=torch.bool, device=device)
        step = 0
        while step < max_latent_steps and active.any():
            # Process only active sequences
            inputs_embeds = current_hidden[active].unsqueeze(1)
            active_past = [pv[active] for pv in past_key_values]
            outputs = self.base_causallm(
                inputs_embeds=inputs_embeds,
                past_key_values=active_past,
                use_cache=True
            )
            # Update past_key_values for active sequences
            for i, pv in enumerate(outputs.past_key_values):
                past_key_values[i][active] = pv
            current_hidden_active = outputs.hidden_states[-1][:, -1, :]

            # Compute logits, sample virtual token, and record log probs
            logits = self.base_causallm.lm_head(current_hidden_active) / temperature
            virtual_token_dist = torch.distributions.Categorical(logits=logits)
            virtual_tokens = virtual_token_dist.sample()
            token_log_prob = virtual_token_dist.log_prob(virtual_tokens)

            # Assign log probs to active sequences
            full_log_prob = torch.full((batch_size,), float('nan'), device=device)
            full_log_prob[active] = token_log_prob
            log_probs.append(full_log_prob)

            # Update active mask: deactivate sequences sampling <|end-latent|>
            end_predicted = virtual_tokens == self.end_id
            active[active.clone()] = ~end_predicted

            # Update hidden states for still-active sequences
            if (~end_predicted).any():
                current_hidden[active] = current_hidden_active[~end_predicted]

            step += 1

        # Generate <|end-latent|> token
        end_token = torch.full((batch_size, 1), self.end_id, device=device)
        tokens = torch.cat([tokens, end_token], dim=1)
        current_attention_mask = torch.cat(
            [current_attention_mask, torch.ones((batch_size, 1), device=device)],
            dim=1
        )

        # Generate answer tokens with log probs
        for _ in range(max_new_tokens):
            outputs = self.base_causallm(
                input_ids=tokens[:, -1:],
                attention_mask=current_attention_mask,
                past_key_values=past_key_values,
                use_cache=True
            )
            next_token_logits = outputs.logits[:, -1, :] / temperature
            next_token_dist = torch.distributions.Categorical(logits=next_token_logits)
            next_token = next_token_dist.sample().unsqueeze(-1)
            token_log_prob = next_token_dist.log_prob(next_token.squeeze(-1))
            log_probs.append(token_log_prob)

            tokens = torch.cat([tokens, next_token], dim=1)
            current_attention_mask = torch.cat(
                [current_attention_mask, torch.ones((batch_size, 1), device=device)],
                dim=1
            )
            past_key_values = outputs.past_key_values

            if (next_token == self.eos_token_id).all():
                break

        if synced_gpus:
            additional_passes = max_new_tokens + max_latent_steps - self.gen_forward_cnt
            while self.gen_forward_cnt < max_new_tokens + max_latent_steps:
                self.gen_forward_cnt += 1
                _ = self.base_causallm(input_ids=tokens[:, -1:], attention_mask=current_attention_mask)

        log_probs = torch.stack(log_probs, dim=1)  # [batch_size, total_steps], with NaN for inactive latent steps
        return tokens, log_probs
    
    def compute_log_probs(self, input_ids, generated_ids):
        """
        Compute log probabilities for a generated sequence
        """
        full_ids = torch.cat([input_ids, generated_ids], dim=1)
        outputs = self.base_causallm(input_ids=full_ids)
        log_probs = torch.nn.functional.log_softmax(outputs.logits, dim=-1)
        shift_log_probs = log_probs[:, :-1, :]
        shift_labels = full_ids[:, 1:]
        token_log_probs = torch.gather(
            shift_log_probs,
            dim=2,
            index=shift_labels.unsqueeze(2)
        ).squeeze(2)
        generated_log_probs = token_log_probs[:, input_ids.shape[1]-1:-1]
        return generated_log_probs
    
    def train(self):
        self.base_causallm.train()
    
    def eval(self):
        self.base_causallm.eval()
