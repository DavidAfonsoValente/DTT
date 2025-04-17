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

    def generate(self, input_ids, attention_mask=None, max_new_tokens=16, max_latent_steps=10, temperature=1.0, **kwargs):
        """Generate 16 completions per GPU for its unique input, distributed across 4 GPUs."""
        if not dist.is_initialized():
            raise RuntimeError("Distributed process group is not initialized.")

        rank = dist.get_rank()
        world_size = dist.get_world_size()
        assert world_size == 4, "This implementation assumes 4 GPUs."

        if rank == 0:
            print(f"[DEBUG] Starting generation with max_new_tokens={max_new_tokens}, max_latent_steps={max_latent_steps}, temperature={temperature}", flush=True)
            print(f"[DEBUG] Input shape: {input_ids.shape}", flush=True)

        batch_size, seq_len = input_ids.shape  # batch_size=1 per GPU
        device = input_ids.device

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, device=device)

        # Each GPU processes its own input (batch_size=1), generating 16 trajectories
        generations_per_gpu = 16
        total_generations = world_size * generations_per_gpu  # 4 * 16 = 64

        # Subset input for this GPU (already handled by DDP data distribution)
        sub_input_ids = input_ids  # [1, seq_len]
        sub_attention_mask = attention_mask  # [1, seq_len]

        # Generate 16 sequences for this GPU's input with sampling
        sub_outputs = self._generate_sub_batch(
            sub_input_ids,
            sub_attention_mask,
            max_new_tokens,
            max_latent_steps,
            temperature,
            generations_per_gpu
        )
        sub_sequences = sub_outputs['sequences']  # List of 16 tensors
        sub_latent_steps = sub_outputs['latent_steps']  # List of 16 ints

        # Gather sequences from all GPUs
        all_sequences_list = [None] * world_size
        dist.all_gather_object(all_sequences_list, sub_sequences)
        all_sequences = [seq for gpu_seqs in all_sequences_list for seq in gpu_seqs]  # Flatten to 64 sequences

        # Move sequences to current device
        all_sequences = [seq.to(device) for seq in all_sequences]

        # Gather latent steps from all GPUs
        all_latent_steps_list = [None] * world_size
        dist.all_gather_object(all_latent_steps_list, sub_latent_steps)
        all_latent_steps = [step for gpu_steps in all_latent_steps_list for step in gpu_steps]  # Flatten to 64 steps

        # Pad sequences to the same length
        max_len = max(seq.size(0) for seq in all_sequences)
        padded_sequences = torch.stack([
            torch.nn.functional.pad(seq, (0, max_len - seq.size(0)), value=self.tokenizer.pad_token_id)
            for seq in all_sequences
        ])

        # Reshape to [world_size * generations_per_gpu, max_len] = [64, max_len]
        padded_sequences = padded_sequences.view(total_generations, -1)

        if rank == 0:
            print(f"[DEBUG] Generated sequences shape: {padded_sequences.shape}", flush=True)

        return {
            'sequences': padded_sequences,  # [64, max_len]
            'latent_steps': all_latent_steps  # List of 64 ints
        }

    def _generate_sub_batch(self, input_ids, attention_mask, max_new_tokens, max_latent_steps, temperature, generations_per_input):
        """Generate multiple completions for a single input with sampling."""
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        sequences = []
        latent_steps_list = []

        for _ in range(generations_per_input):
            sequence = input_ids.clone().squeeze(0)  # [seq_len]
            mode = "token"
            latent_counter = 0
            finished = False
            self.last_hidden_states.append([])  # Per generation
            self.last_logits.append([])  # Per generation
            current_attention_mask = attention_mask.clone().squeeze(0)  # [seq_len]

            outputs = self.base_causallm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            last_hidden_state = outputs.hidden_states[-1][:, -1, :]
            past_key_values = outputs.past_key_values

            for step in range(max_new_tokens):
                if finished:
                    break

                if mode == "token":
                    last_token = sequence[-1].unsqueeze(0).unsqueeze(0)  # [1,1]
                    embed = self.embedding(last_token)
                else:  # latent mode
                    embed = last_hidden_state.unsqueeze(0)  # [1, embedding_dim]

                input_embeds = embed.unsqueeze(0)  # [1,1,embedding_dim]

                current_attention_mask = torch.cat(
                    (current_attention_mask.unsqueeze(0), torch.ones((1, 1), device=device)),  # [1, seq_len] and [1, 1]
                    dim=1
                ).squeeze(0)  # Back to [seq_len + 1]

                outputs = self.base_causallm(
                    inputs_embeds=input_embeds,
                    attention_mask=current_attention_mask.unsqueeze(0),
                    past_key_values=past_key_values,
                    output_hidden_states=True,
                )
                logits = outputs.logits[:, -1, :]  # [1, vocab_size]
                last_hidden_state = outputs.hidden_states[-1][:, -1, :]
                past_key_values = outputs.past_key_values

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

        return {
            'sequences': sequences,
            'latent_steps': latent_steps_list
        }