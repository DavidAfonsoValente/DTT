# DTT.py
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import copy
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from collections import namedtuple
import os

# Set environment variable for synchronous CUDA error reporting
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# Enable anomaly detection for detailed error traces
torch.autograd.set_detect_anomaly(True)

Outputs = namedtuple("Outputs", ["loss", "inputs_embeds", "logits"])

class DTTModel(nn.Module):
    def __init__(self, base_causallm, bot_token_id, eot_token_id, continue_token_id, eos_token_id, tokenizer):
        super(DTTModel, self).__init__()
        self.base_causallm = base_causallm
        self.bot_token_id = bot_token_id
        self.eot_token_id = eot_token_id
        self.continue_token_id = continue_token_id
        self.eos_token_id = eos_token_id
        self.tokenizer = tokenizer  # Store the tokenizer
        self.config = base_causallm.config
        self.name_or_path = base_causallm.config.name_or_path
        self.embedding = base_causallm.get_input_embeddings()
        self.last_hidden_states = []
        self.last_logits = []
        self.warnings_issued = {}
        self._ddp_params_and_buffers_to_ignore = []
        self._model_tags = []
        
        print(f"[DEBUG] DTTModel initialized with config: {self.config}", flush=True)
        print(f"[DEBUG] Special tokens: bot={bot_token_id}, eot={eot_token_id}, continue={continue_token_id}, eos={eos_token_id}", flush=True)

    def __deepcopy__(self, memo):
        print(f"[DEBUG] Creating deep copy of DTTModel", flush=True)
        new_base_causallm = copy.deepcopy(self.base_causallm, memo)
        new_model = DTTModel(
            base_causallm=new_base_causallm,
            bot_token_id=self.bot_token_id,
            eot_token_id=self.eot_token_id,
            continue_token_id=self.continue_token_id,
            eos_token_id=self.eos_token_id,
            tokenizer=self.tokenizer,  # Pass the tokenizer directly
        )
        new_model.last_hidden_states = []
        new_model.last_logits = []
        new_model.warnings_issued = {}
        new_model._ddp_params_and_buffers_to_ignore = []
        print(f"[DEBUG] Deep copy created successfully", flush=True)
        return new_model

    def add_model_tags(self, tags):
        """
        Store tags provided by the trainer.

        Args:
            tags (list): List of tags to associate with the model.
        """
        print(f"[DEBUG] Adding model tags: {tags}", flush=True)
        self._model_tags = tags

    def forward(self, input_ids, attention_mask, labels=None, position_ids=None, **kwargs):
        """
        Forward pass to compute logits and loss, handling latent reasoning with <continue> tokens.

        Args:
            input_ids (torch.Tensor): Input token IDs of shape (batch_size, sequence_length).
            attention_mask (torch.Tensor): Attention mask of shape (batch_size, sequence_length).
            labels (torch.Tensor, optional): Labels for computing loss.
            position_ids (torch.Tensor, optional): Position IDs for the input sequence.
            **kwargs: Additional arguments passed to the base model.

        Returns:
            Outputs: Named tuple containing loss, inputs_embeds, and logits.
        """
        print(f"[DEBUG] Forward pass started with input shape: {input_ids.shape}", flush=True)
        if labels is not None:
            print(f"[DEBUG] Labels provided with shape: {labels.shape}", flush=True)
        
        logits = []

        # Identify positions of <continue> tokens
        continue_indices = (input_ids == self.continue_token_id).nonzero()  # (num_continue_tokens, 2)
        continue_lists = [
            [idx[1].item() for idx in continue_indices if idx[0] == i]
            for i in range(input_ids.shape[0])
        ]  # List per batch item of continue token positions
        
        print(f"[DEBUG] Found {len(continue_indices)} continue tokens", flush=True)
        print(f"[DEBUG] Continue token positions per batch item: {continue_lists}", flush=True)
        
        max_n_continues = max([len(l) for l in continue_lists], default=0)
        print(f"[DEBUG] Max number of continue tokens per batch item: {max_n_continues}", flush=True)

        # Initial compute range and embeddings
        next_compute_range = (0, input_ids.shape[1])
        inputs_embeds = self.embedding(input_ids)
        print(f"[DEBUG] Initial inputs_embeds shape: {inputs_embeds.shape}", flush=True)

        if max_n_continues > 0 and len(continue_indices) > 0:
            next_compute_range = (0, continue_indices[:, 1].min().item())
            print(f"[DEBUG] Adjusting compute range to: {next_compute_range}", flush=True)

        kv_cache = None

        # Process the sequence in segments, handling <continue> tokens
        for pass_idx in range(max_n_continues):
            print(f"[DEBUG] Starting pass {pass_idx+1}/{max_n_continues}", flush=True)
            print(f"[DEBUG] Current compute range: {next_compute_range}", flush=True)
            
            if kv_cache is None:
                print(f"[DEBUG] No KV cache available for this pass", flush=True)
                outputs = self.base_causallm(
                    inputs_embeds=inputs_embeds[:, next_compute_range[0]:next_compute_range[1], :],
                    attention_mask=attention_mask[:, next_compute_range[0]:next_compute_range[1]] if attention_mask is not None else None,
                    position_ids=position_ids[:, next_compute_range[0]:next_compute_range[1]] if position_ids is not None else None,
                    output_hidden_states=True,
                    past_key_values=kv_cache,
                )
                hidden_states_offset = 0
            else:
                print(f"[DEBUG] Using KV cache from previous pass", flush=True)
                past_key_values = [
                    (k[:, :, :next_compute_range[0], :], v[:, :, :next_compute_range[0], :])
                    for k, v in kv_cache
                ]
                outputs = self.base_causallm(
                    inputs_embeds=inputs_embeds[:, next_compute_range[0]:next_compute_range[1], :],
                    attention_mask=attention_mask[:, :next_compute_range[1]] if attention_mask is not None else None,
                    position_ids=position_ids[:, next_compute_range[0]:next_compute_range[1]] if position_ids is not None else None,
                    past_key_values=past_key_values,
                    output_hidden_states=True,
                )
                hidden_states_offset = next_compute_range[0]

            print(f"[DEBUG] Base model output logits shape: {outputs.logits.shape}", flush=True)
            logits.append(outputs.logits)

            # Update the next compute range
            next_compute_range = (
                next_compute_range[1],
                (
                    input_ids.shape[1]
                    if pass_idx + 1 >= max_n_continues
                    else next_compute_range[1] + 1
                ),
            )
            print(f"[DEBUG] Updated compute range for next pass: {next_compute_range}", flush=True)

            hidden_states = outputs.hidden_states[-1]
            print(f"[DEBUG] Hidden states shape: {hidden_states.shape}", flush=True)
            kv_cache = outputs.past_key_values

            # Replace embeddings for <continue> positions with previous hidden states
            filling_indices = [
                (instance_idx, continue_list[pass_idx])
                for instance_idx, continue_list in enumerate(continue_lists)
                if len(continue_list) > pass_idx
            ]
            print(f"[DEBUG] Filling indices for this pass: {filling_indices}", flush=True)
            
            for idx_pair in filling_indices:
                batch_idx, token_idx = idx_pair
                inputs_embeds[batch_idx, token_idx, :] = hidden_states[
                    batch_idx, token_idx - 1 - hidden_states_offset, :
                ]
                print(f"[DEBUG] Replaced embedding at position ({batch_idx}, {token_idx})", flush=True)

        # Final pass for remaining tokens
        print(f"[DEBUG] Performing final pass for remaining tokens", flush=True)
        outputs = self.base_causallm(
            inputs_embeds=inputs_embeds[:, next_compute_range[0]:next_compute_range[1], :],
            attention_mask=attention_mask[:, :next_compute_range[1]] if attention_mask is not None else None,
            position_ids=position_ids[:, next_compute_range[0]:next_compute_range[1]] if position_ids is not None else None,
            past_key_values=(
                [
                    (k[:, :, :next_compute_range[0], :], v[:, :, :next_compute_range[0], :])
                    for k, v in kv_cache
                ]
                if kv_cache
                else None
            ),
            output_hidden_states=True,
        )
        logits.append(outputs.logits)
        print(f"[DEBUG] Final pass output logits shape: {outputs.logits.shape}", flush=True)

        # Concatenate all logits
        logits = torch.cat(logits, dim=-2)
        print(f"[DEBUG] Concatenated logits shape: {logits.shape}", flush=True)

        # Compute loss if labels are provided
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            print(f"[DEBUG] Shifted logits shape: {shift_logits.shape}", flush=True)
            print(f"[DEBUG] Shifted labels shape: {shift_labels.shape}", flush=True)
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            print(f"[DEBUG] Computed loss: {loss.item()}", flush=True)
        else:
            loss = None
            print(f"[DEBUG] No labels provided, skipping loss computation", flush=True)

        print(f"[DEBUG] Forward pass completed", flush=True)
        return Outputs(loss=loss, inputs_embeds=inputs_embeds, logits=logits)

    def generate(self, input_ids, attention_mask=None, max_new_tokens=16, max_latent_steps=50, **kwargs):
        """
        Generate sequences using the model with debugging logs to inspect inputs, outputs, and latent steps.

        Args:
            input_ids (torch.Tensor): Input token IDs of shape [batch_size, seq_len].
            attention_mask (torch.Tensor, optional): Attention mask of shape [batch_size, seq_len].
            max_new_tokens (int): Maximum number of new tokens to generate.
            max_latent_steps (int): Maximum latent steps per sequence in latent mode.
            **kwargs: Additional arguments passed to the base model.

        Returns:
            dict: Dictionary containing:
                - 'sequences': Padded generated token IDs [batch_size, max_len].
                - 'latent_steps': Total latent steps taken per sequence [batch_size].
        """
        # Determine rank for distributed training (default to 0 if not distributed)
        rank = int(os.environ.get("RANK", 0))
        is_rank_zero = rank == 0

        # --- Log Input Details ---
        if is_rank_zero:
            print(f"[DEBUG] Starting generation with max_new_tokens={max_new_tokens}, max_latent_steps={max_latent_steps}")
            print(f"[DEBUG] Input shape: {input_ids.shape}")
            print(f"[DEBUG] Sample input tokens (first sequence): {input_ids[0][:10].tolist()}")
            decoded_input = self.tokenizer.decode(input_ids[0], skip_special_tokens=False)
            print(f"[DEBUG] Decoded input (first sequence): {decoded_input}")
            if attention_mask is not None:
                print(f"[DEBUG] Attention mask shape: {attention_mask.shape}")

        # Extract batch size, sequence length, and device
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Initialize generation state
        sequences = [input_ids[b].clone() for b in range(batch_size)]  # List of sequences
        modes = ["token"] * batch_size  # "token" or "latent" mode per sequence
        latent_steps_counters = [0] * batch_size  # Current latent steps per sequence
        latent_steps_list = [[] for _ in range(batch_size)]  # List of latent step counts per phase
        finished = [False] * batch_size  # Whether each sequence is finished

        # Handle attention mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, device=device)
        current_attention_mask = attention_mask.clone()

        # Compute initial embeddings
        inputs_embeds = self.embedding(input_ids)

        # --- Generation Loop ---
        for step in range(max_new_tokens):
            if is_rank_zero:
                print(f"[DEBUG] Generation step {step+1}/{max_new_tokens}")
                print(f"[DEBUG] Current modes: {modes}")
                print(f"[DEBUG] Latent steps counters: {latent_steps_counters}")
                print(f"[DEBUG] Finished statuses: {finished}")

            # Prepare inputs for the model
            if step == 0:
                current_inputs_embeds = inputs_embeds  # Full initial sequence
            else:
                current_inputs_embeds = new_embeds  # New token or hidden state

            # Forward pass through the base causal language model
            outputs = self.base_causallm(
                inputs_embeds=current_inputs_embeds,
                attention_mask=current_attention_mask,
                past_key_values=None,  # Reprocess entire sequence each step
                output_hidden_states=True,
            )
            logits = outputs.logits[:, -1, :]  # Logits for the last token [batch_size, vocab_size]
            hidden_states = outputs.hidden_states[-1][:, -1, :]  # Last hidden state [batch_size, hidden_size]

            # **Key Modification**: Force the first token to be bot_token_id to start latent mode
            if step == 0:
                next_tokens = torch.full((batch_size,), self.bot_token_id, device=device)
                if is_rank_zero:
                    print(f"[DEBUG] Forced first token to bot_token_id ({self.bot_token_id}) to start latent mode")
            else:
                next_tokens = torch.argmax(logits, dim=-1)  # [batch_size]
                if is_rank_zero:
                    print(f"[DEBUG] Sampled next tokens: {next_tokens.tolist()}")

            # Process each sequence in the batch
            new_embeds_list = []
            for b in range(batch_size):
                if finished[b]:
                    # Append zero embedding for finished sequences
                    new_embeds_list.append(torch.zeros(1, self.embedding.embedding_dim, device=device))
                    continue

                next_token = next_tokens[b].item()
                if modes[b] == "token":
                    # In token mode, check for start of latent reasoning
                    if next_token == self.bot_token_id:
                        modes[b] = "latent"
                        latent_steps_counters[b] = 0
                        if is_rank_zero:
                            print(f"[DEBUG] Sequence {b} switched to latent mode")
                    embed = self.embedding(torch.tensor([next_token], device=device))
                elif modes[b] == "latent":
                    # In latent mode, check for end conditions
                    if next_token == self.eot_token_id or latent_steps_counters[b] >= max_latent_steps:
                        modes[b] = "token"
                        latent_steps_list[b].append(latent_steps_counters[b])
                        if is_rank_zero:
                            print(f"[DEBUG] Sequence {b} switched to token mode after {latent_steps_counters[b]} latent steps")
                        embed = self.embedding(torch.tensor([self.eot_token_id], device=device))
                    else:
                        # Use hidden state as embedding in latent mode
                        embed = hidden_states[b:b+1]
                        latent_steps_counters[b] += 1
                        if is_rank_zero:
                            print(f"[DEBUG] Sequence {b} in latent mode, step {latent_steps_counters[b]}")

                # Update sequence and embeddings
                sequences[b] = torch.cat((sequences[b], next_tokens[b:b+1]))
                new_embeds_list.append(embed)

                # Check for end of sequence
                if next_token == self.eos_token_id:
                    finished[b] = True
                    if is_rank_zero:
                        print(f"[DEBUG] Sequence {b} finished with EOS")
                    if modes[b] == "latent":
                        latent_steps_list[b].append(latent_steps_counters[b])

            # Prepare embeddings for next step
            new_embeds = torch.cat(new_embeds_list, dim=0).unsqueeze(1)
            # Extend attention mask
            current_attention_mask = torch.cat(
                (current_attention_mask, torch.ones(batch_size, 1, device=device)), dim=1
            )

            # Early exit if all sequences are finished
            if all(finished):
                if is_rank_zero:
                    print(f"[DEBUG] All sequences finished at step {step+1}")
                break

        # Compute total latent steps per sequence
        total_latent_steps = [sum(steps) for steps in latent_steps_list]

        # Pad sequences to uniform length
        max_len = max([seq.size(0) for seq in sequences])
        padded_sequences = torch.stack([
            torch.nn.functional.pad(seq, (0, max_len - seq.size(0)), value=self.tokenizer.pad_token_id)
            for seq in sequences
        ])

        # --- Log Output Details ---
        if is_rank_zero:
            generated_text = self.tokenizer.batch_decode(padded_sequences, skip_special_tokens=False)
            print(f"[DEBUG] Actual outputs (generated sequences): {generated_text}")
            print(f"[DEBUG] Total latent steps per sequence: {total_latent_steps}")

        return {
            'sequences': padded_sequences,
            'latent_steps': total_latent_steps
        }