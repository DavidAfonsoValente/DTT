import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import os
import torch.distributed as dist

# Set environment variable for synchronous CUDA error reporting
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# Enable anomaly detection for detailed error traces
torch.autograd.set_detect_anomaly(True)

# Utility function to print memory usage
def print_memory_usage(rank):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        print(f"[rank {rank}] GPU Memory - Allocated: {allocated:.2f} MB, Reserved: {reserved:.2f} MB", flush=True)

class Outputs:
    def __init__(self, loss, inputs_embeds, logits):
        self.loss = loss
        self.inputs_embeds = inputs_embeds
        self.logits = logits

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
        self.epsilon_explore = 0.1  # Probability of adding noise
        self.noise_scale = 0.1     # Scale of Gaussian noise

        print(f"[DEBUG] DTTModel initialized with config: {self.config}", flush=True)
        print(f"[DEBUG] Special tokens: bot={bot_token_id}, eot={eot_token_id}, eos={eos_token_id}", flush=True)

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

    def generate(self, input_ids, attention_mask=None, max_new_tokens=16, max_latent_steps=10, temperature=1.0, generations_per_prompt=4, **kwargs):
        """Generate completions for each prompt in the batch, with batched generation."""
        if not dist.is_initialized():
            raise RuntimeError("Distributed process group is not initialized.")

        rank = dist.get_rank()
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, device=device)

        # Repeat each prompt generations_per_prompt times to create a batch
        repeated_input_ids = input_ids.repeat_interleave(generations_per_prompt, dim=0)
        repeated_attention_mask = attention_mask.repeat_interleave(generations_per_prompt, dim=0)

        print(f"[rank {rank}] Repeated input_ids shape: {repeated_input_ids.shape}", flush=True)

        # Generate all completions in parallel
        outputs = self._generate_sub_batch(
            repeated_input_ids,
            repeated_attention_mask,
            max_new_tokens,
            max_latent_steps,
            temperature
        )

        sequences = outputs['sequences']
        latent_steps = outputs['latent_steps']

        # Convert list of sequences to tensor with padding
        max_len = max(seq.size(0) for seq in sequences)
        padded_sequences = torch.stack([
            torch.nn.functional.pad(seq, (0, max_len - seq.size(0)), value=self.tokenizer.pad_token_id)
            for seq in sequences
        ])

        # Reshape to [batch_size * generations_per_prompt, max_len]
        padded_sequences = padded_sequences.view(batch_size * generations_per_prompt, max_len)

        if rank == 0:
            print(f"[DEBUG] Generated sequences shape: {padded_sequences.shape}", flush=True)

        return {
            'sequences': padded_sequences,
            'latent_steps': latent_steps
        }

    def _generate_sub_batch(self, input_ids, attention_mask, max_new_tokens, max_latent_steps, temperature):
        """Generate multiple completions for a batch of inputs in parallel."""
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        rank = dist.get_rank()

        print(f"[rank {rank}] Starting _generate_sub_batch: input_ids.shape={input_ids.shape}", flush=True)
        print_memory_usage(rank)

        # Initialize sequences with the prompts
        sequences = input_ids.clone()
        current_attention_mask = attention_mask.clone()

        # Initialize states
        modes = torch.zeros(batch_size, device=device, dtype=torch.long)  # 0: token, 1: latent
        finished = torch.zeros(batch_size, device=device, dtype=torch.bool)
        latent_counters = torch.zeros(batch_size, device=device, dtype=torch.long)

        # Initialize past_key_values for efficient generation
        past_key_values = None

        for step in range(max_new_tokens):
            if finished.all():
                break

            # Identify active sequences
            active_indices = ~finished

            # Prepare embeddings for token mode sequences
            token_mode_indices = (modes == 0) & active_indices
            if token_mode_indices.any():
                last_tokens = sequences[token_mode_indices, -1].unsqueeze(1)
                token_embeds = self.embedding(last_tokens)
            else:
                token_embeds = torch.empty((0, 1, self.config.hidden_size), device=device)

            # Prepare embeddings for latent mode sequences
            latent_mode_indices = (modes == 1) & active_indices
            if latent_mode_indices.any() and past_key_values is not None:
                # Use last layer's value states
                last_hidden_states = past_key_values[-1][1][-1, latent_mode_indices, :, -1, :]  # [num_latent_mode, num_heads, head_dim]
                latent_embeds = last_hidden_states.mean(dim=1)  # [num_latent_mode, head_dim]
                # Project to embedding space if dimensions don't match
                if latent_embeds.size(-1) != self.config.hidden_size:
                    latent_embeds = self.base_causallm.lm_head.weight.new_zeros(latent_embeds.size(0), self.config.hidden_size)
            else:
                latent_embeds = torch.empty((0, 1, self.config.hidden_size), device=device)

            # Combine embeddings
            input_embeds = torch.zeros((batch_size, 1, self.config.hidden_size), device=device)
            if token_mode_indices.any():
                input_embeds[token_mode_indices] = token_embeds
            if latent_mode_indices.any():
                input_embeds[latent_mode_indices] = latent_embeds.unsqueeze(1)

            # Extend attention mask
            current_attention_mask = torch.cat(
                (current_attention_mask, torch.ones(batch_size, 1, device=device)),
                dim=1
            )

            # Model forward pass
            outputs = self.base_causallm(
                inputs_embeds=input_embeds,
                attention_mask=current_attention_mask,
                past_key_values=past_key_values,
                output_hidden_states=True,
                return_dict=True
            )
            logits = outputs.logits[:, -1, :]  # [batch_size, vocab_size]
            past_key_values = outputs.past_key_values

            # Process token mode sequences
            if token_mode_indices.any():
                token_logits = logits[token_mode_indices] / temperature
                probs = torch.softmax(token_logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_tokens = torch.tensor([], device=device, dtype=torch.long)

            # Process latent mode sequences
            if latent_mode_indices.any():
                latent_logits = logits[latent_mode_indices]
                virtual_tokens = torch.argmax(latent_logits, dim=-1)
            else:
                virtual_tokens = torch.tensor([], device=device, dtype=torch.long)

            # Update sequences
            next_tokens_full = torch.full((batch_size,), self.tokenizer.pad_token_id, device=device)
            if token_mode_indices.any():
                next_tokens_full[token_mode_indices] = next_tokens

            sequences = torch.cat((sequences, next_tokens_full.unsqueeze(1)), dim=1)

            # Update states
            for i in range(batch_size):
                if finished[i]:
                    continue
                if modes[i] == 0:  # Token mode
                    if next_tokens_full[i] == self.bot_token_id:
                        modes[i] = 1
                        latent_counters[i] = 0
                    elif next_tokens_full[i] == self.eos_token_id:
                        finished[i] = True
                else:  # Latent mode
                    if (latent_mode_indices.any() and virtual_tokens[latent_mode_indices][i] == self.eot_token_id) or \
                       latent_counters[i] >= max_latent_steps:
                        modes[i] = 0
                        sequences[i, -1] = self.eot_token_id
                    else:
                        latent_counters[i] += 1

        # Collect results
        final_sequences = []
        final_latent_steps = []
        for i in range(batch_size):
            seq = sequences[i]
            seq = seq[seq != self.tokenizer.pad_token_id]
            final_sequences.append(seq)
            final_latent_steps.append(latent_counters[i].item())

        print(f"[rank {rank}] Finished _generate_sub_batch: {len(final_sequences)} sequences generated", flush=True)
        print_memory_usage(rank)

        return {
            'sequences': final_sequences,
            'latent_steps': final_latent_steps
        }