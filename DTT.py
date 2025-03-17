# DTT.py
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import copy
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from collections import namedtuple
import sys

Outputs = namedtuple("Outputs", ["loss", "inputs_embeds", "logits"])

class DTTModel(nn.Module):
    def __init__(self, base_causallm, bot_token_id, eot_token_id, continue_token_id, eos_token_id):
        super(DTTModel, self).__init__()
        self.base_causallm = base_causallm
        self.bot_token_id = bot_token_id
        self.eot_token_id = eot_token_id
        self.continue_token_id = continue_token_id
        self.eos_token_id = eos_token_id
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
        Custom generation method implementing DTT inference with latent mode, collecting hidden states and logits.

        Args:
            input_ids (torch.Tensor): Input token IDs of shape (batch_size, sequence_length).
            attention_mask (torch.Tensor, optional): Attention mask for the input.
            max_new_tokens (int): Maximum number of new tokens to generate.
            max_latent_steps (int): Maximum number of latent steps per <bot><eot> pair.
            **kwargs: Additional arguments passed to the base model.

        Returns:
            dict: Dictionary containing:
                - 'sequences': Generated sequences (batch_size, final_length).
                - 'latent_steps': List of latent step counts per sequence.
        """
        print(f"[DEBUG] Starting generation with max_new_tokens={max_new_tokens}, max_latent_steps={max_latent_steps}", flush=True)
        print(f"[DEBUG] Input shape: {input_ids.shape}", flush=True)
        
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        print(f"[DEBUG] Batch size: {batch_size}, Sequence length: {seq_len}, Device: {device}", flush=True)

        # Initialize modes and counters for each sequence
        modes = ["token"] * batch_size
        latent_steps_counters = [0] * batch_size
        latent_steps_list = [[] for _ in range(batch_size)]  # List of latent steps per <bot><eot> pair per sequence
        finished = [False] * batch_size
        sequences = input_ids.clone()
        
        print(f"[DEBUG] Initial modes: {modes}", flush=True)

        # Prepare initial inputs_embeds and attention_mask
        inputs_embeds = self.embedding(input_ids)
        print(f"[DEBUG] Initial inputs_embeds shape: {inputs_embeds.shape}", flush=True)
        
        if attention_mask is None:
            print(f"[DEBUG] No attention mask provided, creating default mask", flush=True)
            attention_mask = torch.ones_like(input_ids, device=device)
        current_attention_mask = attention_mask.clone()
        print(f"[DEBUG] Current attention mask shape: {current_attention_mask.shape}", flush=True)

        # Initialize storage for hidden states and logits
        self.last_hidden_states = [[] for _ in range(batch_size)]
        self.last_logits = [[] for _ in range(batch_size)]

        past_key_values = None

        for step in range(max_new_tokens):
            print(f"[DEBUG] Generation step {step+1}/{max_new_tokens}", flush=True)
            print(f"[DEBUG] Current modes: {modes}", flush=True)
            print(f"[DEBUG] Latent steps counters: {latent_steps_counters}", flush=True)
            print(f"[DEBUG] Finished statuses: {finished}", flush=True)
            
            # Compute outputs
            outputs = self.base_causallm(
                inputs_embeds=inputs_embeds,
                attention_mask=current_attention_mask,
                past_key_values=past_key_values,
                output_hidden_states=True,
            )
            past_key_values = outputs.past_key_values
            logits = outputs.logits[:, -1, :]  # (batch_size, vocab_size)
            hidden_states = outputs.hidden_states[-1][:, -1, :]  # (batch_size, hidden_size)
            
            print(f"[DEBUG] Logits shape: {logits.shape}", flush=True)
            print(f"[DEBUG] Hidden states shape: {hidden_states.shape}", flush=True)

            # Sample next tokens
            next_tokens = torch.argmax(logits, dim=-1)  # (batch_size,)
            print(f"[DEBUG] Sampled next tokens: {next_tokens}", flush=True)

            # Store logits for each generated token
            for b in range(batch_size):
                if not finished[b]:
                    self.last_logits[b].append(logits[b].clone())

            # Process each sequence in the batch
            new_embeds = []
            for b in range(batch_size):
                if finished[b]:
                    # Append padding token embedding for finished sequences
                    new_embeds.append(torch.zeros(1, self.embedding.embedding_dim, device=device))
                    print(f"[DEBUG] Sequence {b} already finished, adding padding", flush=True)
                    continue

                next_token = next_tokens[b].item()
                print(f"[DEBUG] Processing sequence {b}, next token: {next_token}", flush=True)

                if modes[b] == "token":
                    if next_token == self.bot_token_id:
                        modes[b] = "latent"
                        latent_steps_counters[b] = 0
                        print(f"[DEBUG] Sequence {b} switching to latent mode", flush=True)
                    # Append the embedding of the predicted token
                    embed = self.embedding(torch.tensor([next_token], device=device))
                    print(f"[DEBUG] Sequence {b} in token mode, embedding token {next_token}", flush=True)
                elif modes[b] == "latent":
                    if next_token == self.eot_token_id or latent_steps_counters[b] >= max_latent_steps:
                        modes[b] = "token"
                        latent_steps_list[b].append(latent_steps_counters[b])
                        print(f"[DEBUG] Sequence {b} switching to token mode after {latent_steps_counters[b]} latent steps", flush=True)
                        embed = self.embedding(torch.tensor([self.eot_token_id], device=device))
                    else:
                        # Use the hidden state as the next input embedding
                        embed = hidden_states[b:b+1].unsqueeze(0)  # (1, hidden_size)
                        self.last_hidden_states[b].append(hidden_states[b].clone())
                        latent_steps_counters[b] += 1
                        print(f"[DEBUG] Sequence {b} in latent mode, step {latent_steps_counters[b]}, using hidden state", flush=True)

                new_embeds.append(embed)
                sequences[b] = torch.cat((sequences[b], next_tokens[b:b+1]))
                print(f"[DEBUG] Sequence {b} updated, current length: {sequences[b].shape[0]}", flush=True)

                # Check if sequence is finished
                if next_token == self.eos_token_id:
                    finished[b] = True
                    print(f"[DEBUG] Sequence {b} finished with EOS token", flush=True)
                    if modes[b] == "latent":
                        latent_steps_list[b].append(latent_steps_counters[b])
                        print(f"[DEBUG] Added final latent steps count {latent_steps_counters[b]} to sequence {b}", flush=True)

            # Concatenate new embeddings
            new_embeds = torch.cat(new_embeds, dim=0).unsqueeze(1)  # (batch_size, 1, hidden_size)
            print(f"[DEBUG] New embeddings shape: {new_embeds.shape}", flush=True)
            
            inputs_embeds = torch.cat((inputs_embeds, new_embeds), dim=1)
            print(f"[DEBUG] Updated inputs_embeds shape: {inputs_embeds.shape}", flush=True)
            
            current_attention_mask = torch.cat(
                (current_attention_mask, torch.ones(batch_size, 1, device=device)), dim=1
            )
            print(f"[DEBUG] Updated attention mask shape: {current_attention_mask.shape}", flush=True)

            # Check if all sequences are finished
            if all(finished):
                print(f"[DEBUG] All sequences finished, stopping generation at step {step+1}", flush=True)
                break

        # Compute total latent steps per sequence
        total_latent_steps = [sum(steps) for steps in latent_steps_list]
        print(f"[DEBUG] Latent steps per sequence: {latent_steps_list}", flush=True)
        print(f"[DEBUG] Total latent steps per sequence: {total_latent_steps}", flush=True)

        # Convert logits lists to tensors for reward computation
        for b in range(batch_size):
            if self.last_logits[b]:
                self.last_logits[b] = torch.stack(self.last_logits[b])
                print(f"[DEBUG] Sequence {b} logits tensor shape: {self.last_logits[b].shape}", flush=True)
            else:
                self.last_logits[b] = torch.tensor([], device=device)
                print(f"[DEBUG] Sequence {b} has no logits", flush=True)

        print(f"[DEBUG] Generation completed. Final sequences shape: {sequences.shape}", flush=True)
        return {
            'sequences': sequences,
            'latent_steps': total_latent_steps
        }