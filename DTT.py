# DTT.py
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import copy
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn import CrossEntropyLoss
from collections import namedtuple
import os

# Set environment variable for synchronous CUDA error reporting
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# Enable anomaly detection for detailed error traces
torch.autograd.set_detect_anomaly(True)

Outputs = namedtuple("Outputs", ["loss", "inputs_embeds", "logits"])

class DTTModel(nn.Module):
    def __init__(self, base_causallm, bot_token_id, eot_token_id, eos_token_id, tokenizer, num_generations=4):
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
        self.num_generations = num_generations  # Total generations per prompt

        print(f"[DEBUG] DTTModel initialized with config: {self.config}", flush=True)
        print(f"[DEBUG] Special tokens: bot={bot_token_id}, eot={eot_token_id}, eos={eos_token_id}", flush=True)
        print(f"[DEBUG] Number of generations: {self.num_generations}", flush=True)

    def __deepcopy__(self, memo):
        print(f"[DEBUG] Creating deep copy of DTTModel", flush=True)
        new_base_causallm = copy.deepcopy(self.base_causallm, memo)
        new_model = DTTModel(
            base_causallm=new_base_causallm,
            bot_token_id=self.bot_token_id,
            eot_token_id=self.eot_token_id,
            eos_token_id=self.eos_token_id,
            tokenizer=self.tokenizer,
            num_generations=self.num_generations,
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

    def generate(self, input_ids, attention_mask=None, max_new_tokens=16, max_latent_steps=10, **kwargs):
        """Generate completions distributed across multiple GPUs, one sequence per GPU."""
        if not dist.is_initialized():
            raise RuntimeError("Distributed process group is not initialized.")

        rank = dist.get_rank()
        world_size = dist.get_world_size()
        assert world_size == 4, "This implementation assumes 4 GPUs."

        if rank == 0:
            print(f"[DEBUG] Starting generation with max_new_tokens={max_new_tokens}, max_latent_steps={max_latent_steps}", flush=True)
            print(f"[DEBUG] Input shape: {input_ids.shape}", flush=True)

        batch_size, seq_len = input_ids.shape  # batch_size should be 1
        device = input_ids.device

        # Broadcast input_ids and attention_mask from rank 0 to all ranks
        dist.broadcast(input_ids, src=0)
        if attention_mask is not None:
            dist.broadcast(attention_mask, src=0)
        else:
            attention_mask = torch.ones_like(input_ids, device=device)

        # Each GPU generates exactly one sequence
        generations_per_gpu = 1  # Override to 1 sequence per GPU
        total_generations = world_size * generations_per_gpu  # 4 sequences total

        # No need to repeat since each GPU generates one sequence
        sub_input_ids = input_ids  # Shape: [1, seq_len]
        sub_attention_mask = attention_mask  # Shape: [1, seq_len]

        # Print initial memory usage
        print(f"[DEBUG Rank {rank}] Before generation: Memory allocated: {torch.cuda.memory_allocated(device) / 1e9:.2f} GB, Reserved: {torch.cuda.memory_reserved(device) / 1e9:.2f} GB", flush=True)

        sub_outputs = self._generate_sub_batch(sub_input_ids, sub_attention_mask, max_new_tokens, max_latent_steps)

        # Print memory usage after generation
        print(f"[DEBUG Rank {rank}] After generation: Memory allocated: {torch.cuda.memory_allocated(device) / 1e9:.2f} GB, Reserved: {torch.cuda.memory_reserved(device) / 1e9:.2f} GB", flush=True)

        sub_sequences = sub_outputs['sequences']  # List of 1 tensor
        sub_latent_steps = sub_outputs['latent_steps']  # List of 1 int

        # Gather sequences from all GPUs
        all_sequences_list = [None] * world_size
        dist.all_gather_object(all_sequences_list, sub_sequences)
        all_sequences = [seq for gpu_seqs in all_sequences_list for seq in gpu_seqs]  # Flatten to 4 sequences

        # Gather latent steps from all GPUs
        all_latent_steps_list = [None] * world_size
        dist.all_gather_object(all_latent_steps_list, sub_latent_steps)
        all_latent_steps = [step for gpu_steps in all_latent_steps_list for step in gpu_steps]  # Flatten to 4 steps

        # Pad sequences to the same length
        max_len = max(seq.size(0) for seq in all_sequences)
        padded_sequences = torch.stack([
            torch.nn.functional.pad(seq, (0, max_len - seq.size(0)), value=self.tokenizer.pad_token_id)
            for seq in all_sequences
        ])

        # Reshape to [batch_size, num_generations, max_len], where num_generations=4
        padded_sequences = padded_sequences.view(batch_size, total_generations, -1)

        if rank == 0:
            print(f"[DEBUG] Generated sequences shape: {padded_sequences.shape}", flush=True)

        return {
            'sequences': padded_sequences,  # Shape: [1, 4, max_len]
            'latent_steps': all_latent_steps  # List of 4 ints
        }

    def _generate_sub_batch(self, input_ids, attention_mask, max_new_tokens, max_latent_steps):
        """Generate completions for a subset of the batch."""
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        sequences = [input_ids[b].clone() for b in range(batch_size)]
        modes = ["token"] * batch_size
        latent_counters = [0] * batch_size
        finished = [False] * batch_size
        self.last_hidden_states = [[] for _ in range(batch_size)]
        self.last_logits = [[] for _ in range(batch_size)]

        current_attention_mask = attention_mask.clone()

        outputs = self.base_causallm(
            input_ids=input_ids,
            attention_mask=current_attention_mask,
            output_hidden_states=True,
        )
        last_hidden_states = outputs.hidden_states[-1][:, -1, :]
        past_key_values = outputs.past_key_values

        for step in range(max_new_tokens):
            if all(finished):
                break

            input_embeds = []
            for b in range(batch_size):
                if finished[b]:
                    input_embeds.append(torch.zeros(1, self.embedding.embedding_dim, device=device))
                    continue
                if modes[b] == "token":
                    last_token = sequences[b][-1]
                    embed = self.embedding(last_token.unsqueeze(0))
                else:  # latent mode
                    embed = last_hidden_states[b].unsqueeze(0)
                input_embeds.append(embed)
            input_embeds = torch.cat(input_embeds, dim=0).unsqueeze(1)

            current_attention_mask = torch.cat(
                (current_attention_mask, (~torch.tensor(finished, device=device)).int().unsqueeze(1)),
                dim=1
            )

            # Print current sequence lengths and memory usage
            seq_lengths = [seq.size(0) for seq in sequences]
            print(f"[DEBUG Rank {dist.get_rank()}] Step {step}: Sequence lengths: {seq_lengths}, Memory allocated: {torch.cuda.memory_allocated(device) / 1e9:.2f} GB", flush=True)

            outputs = self.base_causallm(
                inputs_embeds=input_embeds,
                attention_mask=current_attention_mask,
                past_key_values=past_key_values,
                output_hidden_states=True,
            )
            logits = outputs.logits[:, -1, :]
            last_hidden_states = outputs.hidden_states[-1][:, -1, :]
            past_key_values = outputs.past_key_values

            # Force the first token to be bot_token_id
            if step == 0:
                logits[:, :] = -float('inf')
                logits[:, self.bot_token_id] = 0

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

        return {
            'sequences': sequences,
            'latent_steps': latent_counters
        }