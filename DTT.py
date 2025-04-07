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
    def __init__(self, base_causallm, bot_token_id, eot_token_id, eos_token_id, tokenizer):
        super(DTTModel, self).__init__()
        self.base_causallm = base_causallm
        self.bot_token_id = bot_token_id  # <start_latent>
        self.eot_token_id = eot_token_id  # <end_latent>
        self.eos_token_id = eos_token_id
        self.tokenizer = tokenizer
        self.config = base_causallm.config
        self.name_or_path = base_causallm.config.name_or_path
        self.embedding = base_causallm.get_input_embeddings()
        self.last_hidden_states = []  # List of lists for latent hidden states per sequence
        self.last_logits = []  # List of lists for latent logits per sequence
        self.warnings_issued = {}
        self._ddp_params_and_buffers_to_ignore = []
        self._model_tags = []
        
        print(f"[DEBUG] DTTModel initialized with config: {self.config}", flush=True)
        print(f"[DEBUG] Special tokens: bot={bot_token_id}, eot={eot_token_id}, eos={eos_token_id}", flush=True)

    def __deepcopy__(self, memo):
        print(f"[DEBUG] Creating deep copy of DTTModel", flush=True)
        new_base_causallm = copy.deepcopy(self.base_causallm, memo)
        new_model = DTTModel(
            base_causallm=new_base_causallm,
            bot_token_id=self.bot_token_id,
            eot_token_id=self.eot_token_id,
            eos_token_id=self.eos_token_id,
            tokenizer=self.tokenizer,
        )
        new_model.last_hidden_states = []
        new_model.last_logits = []
        new_model.warnings_issued = {}
        new_model._ddp_params_and_buffers_to_ignore = []
        print(f"[DEBUG] Deep copy created successfully", flush=True)
        return new_model

    def add_model_tags(self, tags):
        print(f"[DEBUG] Adding model tags: {tags}", flush=True)
        self._model_tags = tags

    def forward(self, input_ids, attention_mask, labels=None, position_ids=None, **kwargs):
        print(f"[DEBUG] Forward pass started with input shape: {input_ids.shape}", flush=True)
        if labels is not None:
            print(f"[DEBUG] Labels provided with shape: {labels.shape}", flush=True)
        
        logits = []

        # Initial compute range and embeddings
        inputs_embeds = self.embedding(input_ids)
        print(f"[DEBUG] Initial inputs_embeds shape: {inputs_embeds.shape}", flush=True)

        outputs = self.base_causallm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=True,
        )
        logits = outputs.logits
        print(f"[DEBUG] Base model output logits shape: {logits.shape}", flush=True)

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
        rank = int(os.environ.get("RANK", 0))
        is_rank_zero = rank == 0

        if is_rank_zero:
            print(f"[DEBUG] Starting generation with max_new_tokens={max_new_tokens}, max_latent_steps={max_latent_steps}")
            print(f"[DEBUG] Input shape: {input_ids.shape}")

        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Initialize state
        sequences = [input_ids[b].clone() for b in range(batch_size)]
        modes = ["token"] * batch_size
        latent_counters = [0] * batch_size
        finished = [False] * batch_size
        self.last_hidden_states = [[] for _ in range(batch_size)]
        self.last_logits = [[] for _ in range(batch_size)]

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, device=device)
        current_attention_mask = attention_mask.clone()

        # Initial forward pass
        outputs = self.base_causallm(
            input_ids=input_ids,
            attention_mask=current_attention_mask,
            output_hidden_states=True,
        )
        last_hidden_states = outputs.hidden_states[-1][:, -1, :]  # [batch_size, hidden_size]
        past_key_values = outputs.past_key_values

        for step in range(max_new_tokens):
            if all(finished):
                break

            # Prepare input embeddings
            input_embeds = []
            for b in range(batch_size):
                if finished[b]:
                    input_embeds.append(torch.zeros(1, self.embedding.embedding_dim, device=device))
                    continue
                if modes[b] == "token":
                    last_token = sequences[b][-1]
                    embed = self.embedding(last_token.unsqueeze(0))
                else:  # latent mode
                    embed = last_hidden_states[b].unsqueeze(0)  # [1, hidden_size]
                input_embeds.append(embed)
            input_embeds = torch.cat(input_embeds, dim=0).unsqueeze(1)  # [batch_size, 1, hidden_size]

            # Extend attention mask
            current_attention_mask = torch.cat(
                (current_attention_mask, (~torch.tensor(finished, device=device)).int().unsqueeze(1)),
                dim=1
            )

            # Forward pass
            outputs = self.base_causallm(
                inputs_embeds=input_embeds,
                attention_mask=current_attention_mask,
                past_key_values=past_key_values,
                output_hidden_states=True,
            )
            logits = outputs.logits[:, -1, :]  # [batch_size, vocab_size]
            last_hidden_states = outputs.hidden_states[-1][:, -1, :]  # [batch_size, hidden_size]
            past_key_values = outputs.past_key_values

            # Process each sequence
            for b in range(batch_size):
                if finished[b]:
                    continue

                if modes[b] == "token":
                    next_token = torch.argmax(logits[b]).unsqueeze(0)
                    sequences[b] = torch.cat((sequences[b], next_token))
                    if next_token.item() == self.bot_token_id:
                        modes[b] = "latent"
                        latent_counters[b] = 0
                    elif next_token.item() == self.eos_token_id:
                        finished[b] = True

                elif modes[b] == "latent":
                    virtual_token = torch.argmax(logits[b]).item()
                    if virtual_token == self.eot_token_id or latent_counters[b] >= max_latent_steps:
                        sequences[b] = torch.cat((sequences[b], torch.tensor([self.eot_token_id], device=device)))
                        modes[b] = "token"
                    else:
                        latent_counters[b] += 1
                        self.last_hidden_states[b].append(last_hidden_states[b].clone())
                        self.last_logits[b].append(logits[b].clone())

            if is_rank_zero:
                print(f"[DEBUG] Step {step}: Modes {modes}, Latent counters {latent_counters}")

        # Pad sequences
        max_len = max(seq.size(0) for seq in sequences)
        padded_sequences = torch.stack([
            torch.nn.functional.pad(seq, (0, max_len - seq.size(0)), value=self.tokenizer.pad_token_id)
            for seq in sequences
        ])

        if is_rank_zero:
            print(f"[DEBUG] Generated sequences shape: {padded_sequences.shape}")

        return {
            'sequences': padded_sequences,
            'latent_steps': latent_counters
        }