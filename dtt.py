"""
Dynamic Thinking Tokens (DTT) Framework

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
            self.max_latent_steps = int(50 - (50 - 20) * progress)
    
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
        
        # Find positions of latent tokens
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
            position_ids=position_ids[:, next_compute_range[0]:next_compute_range[1]],
            past_key_values=past_key_values,
            output_hidden_states=True,
        )
        
        logits.append(outputs.logits)
        hidden_states = outputs.hidden_states[-1]
        hidden_states_list.append(hidden_states)
        
        # Cache key-values for efficiency
        kv_cache = [(k.detach(), v.detach()) for k, v in outputs.past_key_values]
        hidden_states_offset = 0
        
        # Process latent tokens
        for pass_idx in range(max_n_latents):
            # Determine next range to compute
            next_compute_range = (next_compute_range[1], seq_len)
            for i in range(batch_size):
                if len(latent_lists[i]) > pass_idx + 1:
                    next_compute_range = (
                        next_compute_range[0],
                        min(next_compute_range[1], latent_lists[i][pass_idx + 1] + 1),
                    )
            
            if next_compute_range[0] == next_compute_range[1]:
                break
            
            # Update hidden_states_offset
            hidden_states_offset += outputs.past_key_values[0][0].shape[2] - kv_cache[0][0].shape[2]
            
            # Feedback the continuous thoughts to the input_embeds
            filling_indices = [
                (instance_idx, mask_list[pass_idx])
                for instance_idx, mask_list in enumerate(latent_lists)
                if len(mask_list) > pass_idx
            ]
            
            # Break down inputs_embeds to avoid in-place operations
            tensor_list = [
                [
                    inputs_embeds[batch_idx, pos, :]
                    for pos in range(inputs_embeds.shape[1])
                ]
                for batch_idx in range(inputs_embeds.shape[0])
            ]
            
            # Replace latent tokens with continuous thoughts
            for idx_pair in filling_indices:
                batch_idx, token_idx = idx_pair
                # Replace with the preceding last hidden states
                tensor_list[batch_idx][token_idx] = hidden_states[batch_idx, -1, :]
            
            # Reassemble the new inputs_embeds
            inputs_embeds = torch.stack(
                [
                    torch.stack(tensor_list[batch_idx])
                    for batch_idx in range(inputs_embeds.shape[0])
                ]
            )
            
            # Next forward pass
            outputs = self.base_causallm(
                inputs_embeds=inputs_embeds[:, next_compute_range[0]:next_compute_range[1], :],
                attention_mask=attention_mask[:, :next_compute_range[1]],
                position_ids=position_ids[:, next_compute_range[0]:next_compute_range[1]],
                past_key_values=(
                    [
                        (
                            k[:, :, :next_compute_range[0], :],
                            v[:, :, :next_compute_range[0], :],
                        )
                        for k, v in kv_cache
                    ]
                    if kv_cache
                    else None
                ),
                output_hidden_states=True,
            )
            
            logits.append(outputs.logits)
            hidden_states = outputs.hidden_states[-1]
            hidden_states_list.append(hidden_states)
        
        # Final pass
        outputs = self.base_causallm(
            inputs_embeds=inputs_embeds[:, next_compute_range[0]:next_compute_range[1], :],
            attention_mask=attention_mask[:, :next_compute_range[1]],
            position_ids=position_ids[:, next_compute_range[0]:next_compute_range[1]],
            past_key_values=(
                [
                    (
                        k[:, :, :next_compute_range[0], :],
                        v[:, :, :next_compute_range[0], :],
                    )
                    for k, v in kv_cache
                ]
                if kv_cache
                else None
            ),
            output_hidden_states=True,
        )
        
        logits.append(outputs.logits)
        hidden_states_list.append(outputs.hidden_states[-1])
        self.gen_forward_cnt += max_n_latents + 1
        
        # Concatenate logits
        logits = torch.cat(logits, dim=-2)
        
        # Compute loss if labels are provided
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
            
            # Compute advantages (for GRPO)
            if rewards is not None and rewards.shape[0] > 1:
                advantages = self._compute_advantages(rewards)
        
        return Outputs(
            loss=loss,
            inputs_embeds=inputs_embeds,
            logits=logits,
            rewards=rewards,
            advantages=advantages
        )
    
    def _compute_rewards(self, hidden_states_list, logits, labels):
        """
        Compute rewards based on current phase
        
        Components:
        - Binary reward: correctness of final answer
        - CRS (Contrastive Reward Shaping): relative quality signals
        - LCR (Latent Consistency Regularization): coherent trajectories
        - Efficiency reward: penalizes excessive latent steps
        """
        # Placeholder for actual reward computation
        # In a real implementation, this would compute rewards based on
        # correctness, latent space consistency, etc.
        batch_size = hidden_states_list[0].shape[0]
        rewards = torch.zeros(batch_size, device=hidden_states_list[0].device)
        
        # Binary reward component (correctness)
        # This is a placeholder - in real implementation would check answer correctness
        binary_rewards = torch.rand(batch_size, device=rewards.device)
        
        # Latent Consistency Regularization (LCR) - used in warmup phase
        lcr_rewards = torch.zeros_like(rewards)
        if self.w_lcr > 0 and len(hidden_states_list) > 1:
            for i in range(len(hidden_states_list) - 1):
                h1 = hidden_states_list[i]
                h2 = hidden_states_list[i + 1]
                # Compute cosine similarity between consecutive hidden states
                h1_norm = h1 / h1.norm(dim=-1, keepdim=True)
                h2_norm = h2 / h2.norm(dim=-1, keepdim=True)
                cos_sim = (h1_norm * h2_norm).sum(dim=-1).mean(dim=-1)
                lcr_rewards += cos_sim
            lcr_rewards = lcr_rewards / (len(hidden_states_list) - 1)
        
        # Contrastive Reward Shaping (CRS) - used in warmup and core phases
        crs_rewards = torch.zeros_like(rewards)
        if self.w_crs > 0 and batch_size > 1:
            # Simulate CRS with random values for demonstration
            crs_rewards = torch.softmax(binary_rewards, dim=0) - 0.5
        
        # Entropy-Driven Exploration (EDE) - used in core phase
        ede_rewards = torch.zeros_like(rewards)
        if self.w_ede > 0:
            # Simulate entropy rewards with random values
            ede_rewards = torch.rand_like(rewards) * 0.1
        
        # Efficiency reward - penalizes excessive latent steps
        n_latent_steps = len(hidden_states_list) - 1
        efficiency_reward = 0.98 ** n_latent_steps
        
        # Combine rewards based on phase weights
        rewards = (
            self.w_binary * binary_rewards +
            self.w_crs * crs_rewards +
            self.w_lcr * lcr_rewards +
            self.w_ede * ede_rewards +
            0.1 * efficiency_reward  # Fixed weight for efficiency
        )
        
        return rewards
    
    def _compute_advantages(self, rewards):
        """Compute advantages for GRPO"""
        mean_reward = rewards.mean()
        std_reward = rewards.std()
        if std_reward == 0:
            std_reward = 1.0
        advantages = (rewards - mean_reward) / std_reward
        return advantages
    
    def train(self):
        self.base_causallm.train()
    
    def eval(self):
        self.base_causallm.eval()
    
    def generate(
        self,
        input_ids,
        attention_mask,
        max_new_tokens=16,
        output_embedding=False,
        synced_gpus=False,
        **kwargs
    ):
        """
        Generate text with latent reasoning
        """
        self.gen_forward_cnt = 0
        assert input_ids.shape[0] == 1, "only support batch_size == 1 now"
        
        tokens = input_ids[0].detach().tolist()
        labels = input_ids.clone()  # placeholder, not used
        
        outputs = self.forward(
            input_ids,
            torch.ones_like(input_ids, device=input_ids.device),
            labels,
            torch.arange(
                0, input_ids.shape[1], dtype=torch.long, device=input_ids.device
            ).reshape(1, -1),
        )
        
        inputs_embeds = outputs.inputs_embeds
        
        # Get the first token using the current hidden state
        next_token = torch.argmax(outputs.logits[0, -1]).item()
        tokens.append(next_token)
        
        new_token_embed = self.embedding(
            torch.tensor(next_token, device=input_ids.device)
        ).view(1, 1, -1)
        
        new_inputs_embeds = torch.cat((inputs_embeds, new_token_embed), dim=1)
        
        # Get other tokens
        for _ in range(max_new_tokens - 1):
            outputs = self.base_causallm(inputs_embeds=new_inputs_embeds)
            self.gen_forward_cnt += 1
            
            next_token = torch.argmax(outputs.logits[0, -1]).item()
            if next_token == self.eos_token_id:
                break
                
            tokens.append(next_token)
            
            new_token_embed = self.embedding(
                torch.tensor(next_token, device=input_ids.device)
            ).view(1, 1, -1)
            
            new_inputs_embeds = torch.cat((new_inputs_embeds, new_token_embed), dim=1)
        
        if synced_gpus:
            # In FSDP, the number of forward passes needs to be the same across devices
            while self.gen_forward_cnt < max_new_tokens + self.max_latent_steps:
                self.gen_forward_cnt += 1
                _ = self.base_causallm(inputs_embeds=new_inputs_embeds)
        
        if output_embedding:
            # For analysis purpose
            return torch.tensor(tokens).view(1, -1), new_inputs_embeds
        else:
            return torch.tensor(tokens).view(1, -1)
