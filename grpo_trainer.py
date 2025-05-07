"""
GRPO Trainer for DTT - grpo_trainer.py

This module implements the Group Relative Policy Optimization (GRPO) training approach for DTT.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

import os
import math
import time
import random
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Union, Tuple

class GRPOTrainer:
    """
    Group Relative Policy Optimization Trainer for DTT
    """
    
    def __init__(
        self,
        model,
        ref_model,
        optimizer,
        tokenizer,
        configs,
        device,
        rank=0,
        world_size=1,
    ):
        self.model = model
        self.ref_model = ref_model
        self.optimizer = optimizer
        self.tokenizer = tokenizer
        self.configs = configs
        self.device = device
        self.rank = rank
        self.world_size = world_size
        
        self.kl_coef = configs.kl_coef if hasattr(configs, 'kl_coef') else 0.04
        self.clip_range = configs.clip_range if hasattr(configs, 'clip_range') else 0.2
        self.group_size = configs.group_size if hasattr(configs, 'group_size') else 8
        
        self.total_steps = 0
        self.best_reward = float('-inf')
        
        self.total_training_steps = configs.num_epochs * configs.steps_per_epoch
        self.warmup_steps = int(0.2 * self.total_training_steps)
        self.core_steps = int(0.7 * self.total_training_steps)
        
        self._update_phase(0)
    
    def _update_phase(self, step):
        """Update training phase based on current step"""
        if step < self.warmup_steps:
            phase = "warmup"
        elif step < self.warmup_steps + self.core_steps:
            phase = "core"
            core_progress = (step - self.warmup_steps) / self.core_steps
            self.model.module.set_phase(phase, core_progress, 1.0)
            return
        else:
            phase = "final"
        
        self.model.module.set_phase(phase)
    
    def train_step(self, batch, grad_accumulation_steps=1):
        """
        Perform a single GRPO training step with generation
        """
        self.model.train()
        self.ref_model.eval()
        
        self._update_phase(self.total_steps)
        
        input_ids = batch['input_ids'].to(self.device)
        attention_masks = batch['attention_mask'].to(self.device)
        answers = batch['answer']
        batch_size = input_ids.shape[0]
        
        total_loss = 0
        metrics = {
            'loss': 0,
            'kl': 0,
            'reward': 0,
            'policy_ratio': 0,
            'advantage': 0,
        }
        
        for i in range(batch_size):
            # Repeat input for group_size sequences
            group_input_ids = input_ids[i:i+1].repeat(self.group_size, 1)
            group_attention_masks = attention_masks[i:i+1].repeat(self.group_size, 1)
            answer = answers[i]
            
            # Generate multiple sequences
            generated_sequences, log_probs = self.model.module.generate_with_log_probs(
                input_ids=group_input_ids,
                attention_mask=group_attention_masks,
                max_new_tokens=64,
                num_latent_steps=self.model.module.max_latent_steps
            )
            
            # Compute rewards
            rewards = self._compute_rewards_from_outputs(generated_sequences, answer)
            advantages = self._compute_advantages(rewards)
            
            # Compute reference log probs
            with torch.no_grad():
                ref_log_probs = self.ref_model.module.compute_log_probs(
                    group_input_ids,
                    generated_sequences[:, input_ids.shape[1]:]
                )
            
            # Policy loss
            ratio = torch.exp(log_probs.sum(dim=1) - ref_log_probs.sum(dim=1))
            policy_loss_1 = ratio * advantages
            policy_loss_2 = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * advantages
            policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()
            
            kl = (log_probs.sum(dim=1) - ref_log_probs.sum(dim=1)).mean()
            loss = policy_loss + self.kl_coef * kl
            loss = loss / grad_accumulation_steps
            
            loss.backward()
            
            total_loss += loss.item() * grad_accumulation_steps
            metrics['loss'] += loss.item() * grad_accumulation_steps
            metrics['kl'] += kl.item()
            metrics['reward'] += rewards.mean().item()
            metrics['policy_ratio'] += ratio.mean().item()
            metrics['advantage'] += advantages.mean().item()
        
        for k in metrics:
            metrics[k] /= batch_size
        
        self.total_steps += 1
        return metrics
    
    def _compute_rewards_from_outputs(self, generated_sequences, ground_truth_answer):
        """
        Compute rewards based on generated outputs vs ground truth
        """
        rewards = []
        for seq in generated_sequences:
            generated_text = self.tokenizer.decode(seq, skip_special_tokens=True)
            if '<|end-latent|>' in generated_text:
                answer_part = generated_text.split('<|end-latent|>')[1].strip()
                reward = 1 if answer_part == ground_truth_answer else 0
            else:
                reward = 0
            rewards.append(reward)
        return torch.tensor(rewards, device=self.device)
    
    def _compute_advantages(self, rewards):
        mean_reward = rewards.mean()
        std_reward = rewards.std() + 1e-8
        advantages = (rewards - mean_reward) / std_reward
        return advantages
    
    def evaluate(self, eval_dataloader):
        """
        Evaluate the model on validation data
        """
        self.model.eval()
        total_reward = 0
        num_samples = 0
        
        with torch.no_grad():
            for batch in eval_dataloader:
                batch = {k: v.to(self.device) for k, v in batch.items() if k != 'idx'}
                input_ids = batch['input_ids']
                attention_masks = batch['attention_mask']
                answers = batch['answer']
                batch_size = input_ids.shape[0]
                
                for i in range(batch_size):
                    gen_ids = self.model.module.generate(
                        input_ids=input_ids[i:i+1],
                        attention_mask=attention_masks[i:i+1],
                        max_new_tokens=64,
                        num_latent_steps=self.model.module.max_latent_steps
                    )
                    rewards = self._compute_rewards_from_outputs(gen_ids, answers[i])
                    total_reward += rewards.mean().item()
                    num_samples += 1
        
        metrics = {
            'eval_reward': total_reward / num_samples if num_samples > 0 else 0,
        }
        
        return metrics
    
    def save_checkpoint(self, path, epoch, metrics=None):
        if self.rank == 0:
            checkpoint = {
                'model': self.model.module.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'epoch': epoch,
                'total_steps': self.total_steps,
                'metrics': metrics,
                'configs': self.configs,
            }
            torch.save(checkpoint, path)
            print(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path):
        if not os.path.exists(path):
            return 0
        
        checkpoint = torch.load(path, map_location=self.device)
        self.model.module.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.total_steps = checkpoint['total_steps']
        
        self._update_phase(self.total_steps)
        
        return checkpoint['epoch']