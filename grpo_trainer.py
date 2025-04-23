"""
GRPO Trainer for Dynamic Thinking Tokens (DTT) Framework

This module implements the Group Relative Policy Optimization (GRPO) training
approach for the DTT framework, extending Coconut with reinforcement learning.
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
    
    Implements the GRPO algorithm for training DTT models with reinforcement learning.
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
        """
        Initialize the GRPO trainer
        
        Args:
            model: The DTT model to train
            ref_model: Reference model for KL penalty calculation
            optimizer: Optimizer for model updates
            tokenizer: Tokenizer for processing text
            configs: Configuration parameters
            device: Device to run training on
            rank: Process rank for distributed training
            world_size: Total number of processes for distributed training
        """
        self.model = model
        self.ref_model = ref_model
        self.optimizer = optimizer
        self.tokenizer = tokenizer
        self.configs = configs
        self.device = device
        self.rank = rank
        self.world_size = world_size
        
        # GRPO hyperparameters
        self.kl_coef = configs.kl_coef if hasattr(configs, 'kl_coef') else 0.04
        self.clip_range = configs.clip_range if hasattr(configs, 'clip_range') else 0.2
        self.group_size = configs.group_size if hasattr(configs, 'group_size') else 8
        
        # Training state
        self.total_steps = 0
        self.best_reward = float('-inf')
        
        # Phase management
        self.total_training_steps = configs.num_epochs * configs.steps_per_epoch
        self.warmup_steps = int(0.2 * self.total_training_steps)
        self.core_steps = int(0.7 * self.total_training_steps)
        
        # Set initial phase
        self._update_phase(0)
    
    def _update_phase(self, step):
        """Update training phase based on current step"""
        if step < self.warmup_steps:
            phase = "warmup"
        elif step < self.warmup_steps + self.core_steps:
            phase = "core"
            # Calculate progress within core phase for annealing
            core_progress = (step - self.warmup_steps) / self.core_steps
            self.model.module.set_phase(phase, core_progress, 1.0)
            return
        else:
            phase = "final"
        
        self.model.module.set_phase(phase)
    
    def train_step(self, batch, grad_accumulation_steps=1):
        """
        Perform a single GRPO training step
        
        Args:
            batch: Batch of data containing multiple groups
            grad_accumulation_steps: Number of steps to accumulate gradients
            
        Returns:
            dict: Training metrics
        """
        self.model.train()
        self.ref_model.eval()
        
        # Update phase based on current step
        self._update_phase(self.total_steps)
        
        # Process batch into groups
        groups = self._prepare_groups(batch)
        total_loss = 0
        metrics = {
            'loss': 0,
            'kl': 0,
            'reward': 0,
            'policy_ratio': 0,
            'advantage': 0,
        }
        
        # Process each group
        for group_idx, group in enumerate(groups):
            # Forward pass with reward computation
            outputs = self.model(
                **group,
                compute_rewards=True,
            )
            
            rewards = outputs.rewards
            advantages = outputs.advantages
            
            # Get log probs from current policy
            logits = outputs.logits
            log_probs = self._get_log_probs(logits, group['labels'])
            
            # Get log probs from reference policy
            with torch.no_grad():
                ref_outputs = self.ref_model(**group)
                ref_logits = ref_outputs.logits
                ref_log_probs = self._get_log_probs(ref_logits, group['labels'])
            
            # Calculate policy ratio and clipped objective
            ratio = torch.exp(log_probs - ref_log_probs)
            policy_loss_1 = ratio * advantages.unsqueeze(-1)
            policy_loss_2 = torch.clamp(
                ratio,
                1.0 - self.clip_range,
                1.0 + self.clip_range
            ) * advantages.unsqueeze(-1)
            policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()
            
            # Calculate KL divergence for penalty
            kl = (ref_log_probs - log_probs).mean()
            
            # Combined loss
            loss = policy_loss + self.kl_coef * kl
            loss = loss / grad_accumulation_steps
            
            # Backward pass
            loss.backward()
            
            # Update metrics
            total_loss += loss.item() * grad_accumulation_steps
            metrics['loss'] += loss.item() * grad_accumulation_steps
            metrics['kl'] += kl.item()
            metrics['reward'] += rewards.mean().item()
            metrics['policy_ratio'] += ratio.mean().item()
            metrics['advantage'] += advantages.mean().item()
        
        # Average metrics across groups
        for k in metrics:
            metrics[k] /= len(groups)
        
        self.total_steps += 1
        return metrics
    
    def _prepare_groups(self, batch):
        """
        Prepare batch data into groups for GRPO
        
        Args:
            batch: Batch of data
            
        Returns:
            list: List of group data dictionaries
        """
        # This is a simplified implementation
        # In practice, you would create multiple variations/samples for each input
        
        batch_size = batch['input_ids'].shape[0]
        groups = []
        
        # Split batch into groups
        for i in range(0, batch_size, self.group_size):
            end_idx = min(i + self.group_size, batch_size)
            group = {
                'input_ids': batch['input_ids'][i:end_idx].to(self.device),
                'attention_mask': batch['attention_mask'][i:end_idx].to(self.device),
                'labels': batch['labels'][i:end_idx].to(self.device),
                'position_ids': batch['position_ids'][i:end_idx].to(self.device) if 'position_ids' in batch else None,
            }
            groups.append(group)
        
        return groups
    
    def _get_log_probs(self, logits, labels):
        """
        Calculate log probabilities from logits and labels
        
        Args:
            logits: Model logits
            labels: Target labels
            
        Returns:
            torch.Tensor: Log probabilities
        """
        # Shift logits and labels for next-token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # Calculate log probabilities
        log_probs = F.log_softmax(shift_logits, dim=-1)
        
        # Gather the log probs at the positions of the labels
        label_mask = (shift_labels >= 0).float()
        gathered_log_probs = torch.gather(
            log_probs,
            dim=-1,
            index=torch.clamp(shift_labels, min=0).unsqueeze(-1)
        ).squeeze(-1)
        
        # Apply mask for padding tokens
        masked_log_probs = gathered_log_probs * label_mask
        
        # Sum log probs over sequence dimension
        seq_lengths = label_mask.sum(dim=-1, keepdim=True)
        seq_lengths = torch.clamp(seq_lengths, min=1.0)  # Avoid division by zero
        
        # Average log probs over sequence length
        token_log_probs = masked_log_probs.sum(dim=-1, keepdim=True) / seq_lengths
        
        return token_log_probs
    
    def evaluate(self, eval_dataloader):
        """
        Evaluate the model on validation data
        
        Args:
            eval_dataloader: DataLoader for evaluation data
            
        Returns:
            dict: Evaluation metrics
        """
        self.model.eval()
        total_loss = 0
        total_reward = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in eval_dataloader:
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items() if k != 'idx'}
                
                # Forward pass with reward computation
                outputs = self.model(
                    **batch,
                    compute_rewards=True,
                )
                
                loss = outputs.loss
                rewards = outputs.rewards
                
                # Gather metrics across distributed processes
                if self.world_size > 1:
                    dist.all_reduce(loss, op=dist.ReduceOp.SUM)
                    dist.all_reduce(rewards.mean(), op=dist.ReduceOp.SUM)
                    loss = loss / self.world_size
                    rewards = rewards / self.world_size
                
                total_loss += loss.item()
                total_reward += rewards.mean().item()
                num_batches += 1
        
        metrics = {
            'eval_loss': total_loss / num_batches,
            'eval_reward': total_reward / num_batches,
        }
        
        return metrics
    
    def save_checkpoint(self, path, epoch, metrics=None):
        """
        Save model checkpoint
        
        Args:
            path: Path to save checkpoint
            epoch: Current epoch
            metrics: Optional evaluation metrics
        """
        if self.rank == 0:
            # Only save from rank 0 in distributed training
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
        """
        Load model checkpoint
        
        Args:
            path: Path to checkpoint
            
        Returns:
            int: Last completed epoch
        """
        if not os.path.exists(path):
            return 0
        
        checkpoint = torch.load(path, map_location=self.device)
        self.model.module.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.total_steps = checkpoint['total_steps']
        
        # Update phase based on loaded step count
        self._update_phase(self.total_steps)
        
        return checkpoint['epoch']
