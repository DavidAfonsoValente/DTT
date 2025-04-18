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
        """Generate completions for each prompt in the batch, distributed across GPUs."""
        if not dist.is_initialized():
            raise RuntimeError("Distributed process group is not initialized.")

        rank = dist.get_rank()

        if rank == 0:
            print(f"[DEBUG] Starting generation with max_new_tokens={max_new_tokens}, max_latent_steps={max_latent_steps}, temperature={temperature}", flush=True)
            print(f"[DEBUG] Input shape: {input_ids.shape}", flush=True)

        batch_size, seq_len = input_ids.shape  # batch_size can be > 1 (e.g., 4)
        device = input_ids.device
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, device=device)

        all_sequences = []
        all_latent_steps = []

        # Process each prompt in the batch individually
        for b in range(batch_size):
            sub_input_ids = input_ids[b:b+1]  # [1, seq_len]
            sub_attention_mask = attention_mask[b:b+1]  # [1, seq_len]
            sub_outputs = self._generate_sub_batch(
                sub_input_ids,
                sub_attention_mask,
                max_new_tokens,
                max_latent_steps,
                temperature,
                generations_per_input=generations_per_prompt  
            )
            all_sequences.extend(sub_outputs['sequences'])
            all_latent_steps.extend(sub_outputs['latent_steps'])

        # Pad sequences to the same length
        max_len = max(seq.size(0) for seq in all_sequences)
        padded_sequences = torch.stack([
            torch.nn.functional.pad(seq, (0, max_len - seq.size(0)), value=self.tokenizer.pad_token_id)
            for seq in all_sequences
        ])

        # Reshape to [batch_size * generations_per_prompt, max_len]
        padded_sequences = padded_sequences.view(batch_size * generations_per_prompt, -1)

        if rank == 0:
            print(f"[DEBUG] Generated sequences shape: {padded_sequences.shape}", flush=True)

        return {
            'sequences': padded_sequences,  # [batch_size * 4, max_len], e.g., [16, max_len] for batch_size=4
            'latent_steps': all_latent_steps  # List of batch_size * 4 ints
        }

    def _generate_sub_batch(self, input_ids, attention_mask, max_new_tokens, max_latent_steps, temperature, generations_per_input):
        """Generate multiple completions for a single input with sampling."""
        batch_size, seq_len = input_ids.shape  # Should be [1, seq_len]
        assert batch_size == 1, "Expected batch_size=1 for sub_batch"
        device = input_ids.device
        sequences = []
        latent_steps_list = []
        rank = self.accelerator.local_rank if hasattr(self, 'accelerator') else dist.get_rank()  # Fallback to dist.get_rank()

        print(f"[rank {rank}] Starting _generate_sub_batch: input_ids.shape={input_ids.shape}, seq_len={seq_len}", flush=True)
        print_memory_usage(rank)

        for gen_idx in range(generations_per_input):
            print(f"[rank {rank}] Generating completion {gen_idx + 1}/{generations_per_input}", flush=True)
            sequence = input_ids.clone().squeeze(0)  # [seq_len]
            mode = "token"
            latent_counter = 0
            finished = False
            self.last_hidden_states.append([])  # Per generation
            self.last_logits.append([])  # Per generation
            current_attention_mask = attention_mask.clone().squeeze(0)  # [seq_len]
            print_memory_usage(rank)

            # Initial forward pass
            print(f"[rank {rank}] Initial forward pass for prompt", flush=True)
            outputs = self.base_causallm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            last_hidden_state = outputs.hidden_states[-1][:, -1, :]
            past_key_values = outputs.past_key_values
            del outputs  # Free memory immediately

            for step in range(max_new_tokens):
                if finished:
                    break

                if mode == "token":
                    last_token = sequence[-1].unsqueeze(0).unsqueeze(0)  # [1,1]
                    embed = self.embedding(last_token)
                else:  # latent mode
                    embed = last_hidden_state.unsqueeze(0)  # [1, embedding_dim]
                    if torch.rand(1, device=device).item() < self.epsilon_explore:
                        noise = torch.randn_like(embed) * self.noise_scale
                        embed = embed + noise

                input_embeds = embed  # [1,1,hidden_size]

                # Extend attention mask
                current_attention_mask = torch.cat(
                    (current_attention_mask, torch.ones(1, device=device)),
                    dim=0
                )

                # Forward pass
                outputs = self.base_causallm(
                    inputs_embeds=input_embeds,
                    attention_mask=current_attention_mask.unsqueeze(0),  # [1, seq_len + step + 1]
                    past_key_values=past_key_values,
                    output_hidden_states=True,
                )
                logits = outputs.logits[:, -1, :]  # [1, vocab_size]
                last_hidden_state = outputs.hidden_states[-1][:, -1, :]
                past_key_values = outputs.past_key_values
                del outputs  # Free memory after extraction

                if step == 0:
                    # Force first token to be bot_token_id
                    next_token = torch.tensor([self.bot_token_id], device=device)
                else:
                    if mode == "token":
                        logits = logits / temperature
                        probs = torch.softmax(logits, dim=-1)
                        next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
                    else:
                        virtual_token = torch.argmax(logits).item()
                        if virtual_token == self.eot_token_id or latent_counter >= max_latent_steps:
                            next_token = torch.tensor([self.eot_token_id], device=device)
                            mode = "token"
                        else:
                            latent_counter += 1
                            # Store hidden states selectively (e.g., every other step)
                            if latent_counter % 2 == 0:
                                self.last_hidden_states[-1].append(last_hidden_state.clone())
                                self.last_logits[-1].append(logits.clone())
                            continue

                sequence = torch.cat((sequence, next_token), dim=0)
                if mode == "token":
                    if next_token.item() == self.bot_token_id:
                        mode = "latent"
                        latent_counter = 0
                    elif next_token.item() == self.eos_token_id:
                        finished = True

            sequences.append(sequence)
            latent_steps_list.append(latent_counter)
            print(f"[rank {rank}] Completion {gen_idx + 1} finished: sequence.shape={sequence.shape}, latent_steps={latent_counter}", flush=True)
            print_memory_usage(rank)

            # Clear memory after each completion
            del sequence, last_hidden_state, past_key_values
            torch.cuda.empty_cache()

        print(f"[rank {rank}] Finished _generate_sub_batch: {len(sequences)} sequences generated", flush=True)
        print_memory_usage(rank)

        return {
            'sequences': sequences,
            'latent_steps': latent_steps_list
        }

def print_memory_usage(rank):
    allocated = torch.cuda.memory_allocated(rank) / (1024 ** 3)  # in GB
    reserved = torch.cuda.memory_reserved(rank) / (1024 ** 3)    # in GB
    print(f"[rank {rank}] Memory Allocated: {allocated:.3f} GB, Reserved: {reserved:.3f} GB", flush=True)