import os
import sys
import yaml
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig
from model import SparseGatedModel
from reward import custom_reward
from utils import preprocess_dataset
import wandb
from typing import Dict, Any
from pathlib import Path

def setup_ddp() -> tuple[int, int, int]:
    """Initialize DistributedDataParallel for multi-GPU training."""
    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    return local_rank, global_rank, world_size

def cleanup_ddp() -> None:
    """Clean up DDP environment."""
    destroy_process_group()

@torch.no_grad()
def generate_completions(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    max_new_tokens: int,
    temperature: float,
    do_sample: bool,
    pad_token_id: int,
    eos_token_id: int,
    gumbel_hard: bool
) -> tuple[torch.Tensor, list[list[float]], list[list[float]], torch.Tensor]:
    """
    Generate completions and collect old log probabilities.
    """
    batch_size, seq_len = input_ids.shape
    device = input_ids.device
    generated_ids = input_ids.clone()
    current_attention_mask = attention_mask.clone()
    gate_values = [[] for _ in range(batch_size)]
    old_log_probs = [[] for _ in range(batch_size)]
    still_generating = torch.ones(batch_size, device=device, dtype=torch.bool)

    outputs, last_hidden = model(
        input_ids=input_ids,
        attention_mask=current_attention_mask,
        use_cache=True,
        gumbel_hard_during_forward=gumbel_hard,
        return_last_hidden_state=True
    )
    past_key_values = outputs.past_key_values
    next_hidden = last_hidden[:, -1:, :]
    next_logits = outputs.logits[:, -1, :]

    for _ in range(max_new_tokens):
        if do_sample:
            probs = torch.softmax(next_logits / temperature, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1)
        else:
            next_tokens = torch.argmax(next_logits, dim=-1).unsqueeze(-1)

        log_probs = torch.log_softmax(next_logits, dim=-1)
        chosen_log_prob = log_probs.gather(1, next_tokens).squeeze(1)
        for i in range(batch_size):
            if still_generating[i]:
                old_log_probs[i].append(chosen_log_prob[i].item())

        generated_ids = torch.cat([generated_ids, next_tokens], dim=1)
        current_attention_mask = torch.cat([current_attention_mask, torch.ones_like(next_tokens, device=device)], dim=1)

        eos_mask = (next_tokens.squeeze(-1) == eos_token_id)
        still_generating = still_generating & (~eos_mask)

        if not still_generating.any():
            break

        step_outputs, step_hidden = model(
            input_ids=next_tokens,
            attention_mask=current_attention_mask,
            past_key_values=past_key_values,
            prev_step_last_hidden_state=next_hidden,
            use_cache=True,
            gumbel_hard_during_forward=gumbel_hard,
            return_last_hidden_state=True
        )

        step_gates = model.module.current_gate_values_for_batch.squeeze()
        for i in range(batch_size):
            if still_generating[i]:
                gate_values[i].append(step_gates[i].item() if batch_size > 1 else step_gates.item())

        past_key_values = step_outputs.past_key_values
        next_hidden = step_hidden
        next_logits = step_outputs.logits[:, -1, :]

    return generated_ids, gate_values, old_log_probs, current_attention_mask

def compute_log_probs(model: torch.nn.Module, sequences: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Compute per-token log probabilities for sequences.
    """
    outputs = model(input_ids=sequences, attention_mask=attention_mask)
    logits = outputs.logits
    log_probs = F.log_softmax(logits, dim=-1)
    indices = sequences.unsqueeze(-1).expand(-1, -1, log_probs.size(-1))
    selected_log_probs = log_probs.gather(2, indices).squeeze(2)
    return selected_log_probs

def pad_sequences(sequences: list[torch.Tensor], pad_value: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pad sequences to the maximum length on the right.
    """
    max_len = max(len(seq) for seq in sequences)
    padded_sequences = [
        torch.cat([seq, torch.full((max_len - len(seq),), pad_value, dtype=seq.dtype, device=seq.device)])
        for seq in sequences
    ]
    attention_masks = [
        torch.cat([torch.ones(len(seq), dtype=torch.long, device=seq.device),
                   torch.zeros(max_len - len(seq), dtype=torch.long, device=seq.device)])
        for seq in sequences
    ]
    return torch.stack(padded_sequences), torch.stack(attention_masks)

def compute_grpo_loss(new_log_probs: torch.Tensor, old_log_probs: torch.Tensor, advantages: torch.Tensor, mask: torch.Tensor, epsilon_clip: float, beta_kl: float = 0.0) -> torch.Tensor:
    """
    Compute GRPO loss with clipping and masking.
    """
    ratios = torch.exp(new_log_probs - old_log_probs)
    surrogate1 = ratios * advantages
    clipped_ratios = torch.clamp(ratios, 1 - epsilon_clip, 1 + epsilon_clip)
    surrogate2 = clipped_ratios * advantages
    loss = -torch.min(surrogate1, surrogate2)
    masked_loss = loss * mask
    loss_value = masked_loss.sum() / mask.sum()
    if beta_kl > 0:
        kl_div = (new_log_probs - old_log_probs).sum(dim=1).mean()
        loss_value += beta_kl * kl_div
    return loss_value

def main(config_path: str) -> None:
    """Main training function with GRPO implementation."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    local_rank, global_rank, world_size = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")

    if global_rank == 0:
        wandb.init(project=config.get("wandb_project", "grpo-training"), name=config["output_dir"], config=config)
        wandb.save(config_path)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        config["model_name"],
        quantization_config=bnb_config,
        device_map={"": device}
    )

    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    lora_config = LoraConfig(
        r=config["lora_rank"],
        lora_alpha=config["lora_rank"] * 2,
        target_modules=config.get("lora_target_modules"),
        lora_dropout=0.05,
        bias="none",
    )

    model = SparseGatedModel(
        model=base_model,
        peft_config=lora_config,
        hidden_size=config["hidden_size"],
        embedding_dim=base_model.config.hidden_size,
        gate_temperature=config["initial_gate_temperature"],
    ).to(device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    train_dataset = preprocess_dataset(
        config["dataset_name"], config["data_dir"], tokenizer, config["max_prompt_length"], "train"
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["per_device_train_batch_size"],
        sampler=train_sampler,
        num_workers=4
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"])
    num_training_steps = (
        len(train_dataset) // (config["per_device_train_batch_size"] * world_size * config["gradient_accumulation_steps"])
    ) * config["num_train_epochs"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_training_steps,
        eta_min=0
    )

    global_step = 0
    for epoch in range(config["num_train_epochs"]):
        train_sampler.set_epoch(epoch)
        model.train()

        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            ground_truths = batch["ground_truths"]

            # Compute per-sequence prompt lengths
            prompt_lens = attention_mask.sum(dim=1)

            expanded_input_ids = input_ids.repeat_interleave(config["group_size_grpo"], dim=0)
            expanded_attention_mask = attention_mask.repeat_interleave(config["group_size_grpo"], dim=0)

            generated_ids, gate_values, old_log_probs, _ = generate_completions(
                model,
                expanded_input_ids,
                expanded_attention_mask,
                config["max_completion_length"],
                config["sampling_temperature_grpo"],
                config["sampling_temperature_grpo"] > 0,
                tokenizer.pad_token_id,
                tokenizer.eos_token_id,
                config["gumbel_hard_generation_train"]
            )

            completions_ids = [s.tolist() for s in generated_ids]
            rewards = custom_reward(
                completions_ids,
                gate_values,
                tokenizer,
                ground_truths,
                prompt_lens,
                config["lambda_penalty"],
                config["gate_penalty_coeff"],
                config["group_size_grpo"]
            )

            rewards_tensor = torch.tensor(rewards, device=device).view(-1, config["group_size_grpo"])
            mean_rewards = rewards_tensor.mean(dim=1, keepdim=True)
            std_rewards = rewards_tensor.std(dim=1, keepdim=True)
            advantages = (rewards_tensor - mean_rewards) / (std_rewards + 1e-8)

            padded_sequences, padded_masks = pad_sequences(generated_ids, tokenizer.pad_token_id)
            new_log_probs = compute_log_probs(model, padded_sequences, padded_masks)

            # Align old_log_probs with new_log_probs
            old_log_probs_tensor = []
            for i in range(len(generated_ids)):
                seq_len = len(generated_ids[i])
                old_log_probs_padded = old_log_probs[i] + [0.0] * (seq_len - len(old_log_probs[i]))
                old_log_probs_tensor.append(torch.tensor(old_log_probs_padded, device=device))
            old_log_probs_tensor = torch.stack(old_log_probs_tensor)

            # Create mask for generated tokens
            mask = torch.zeros_like(new_log_probs)
            for i in range(len(generated_ids)):
                prompt_len = prompt_lens[i // config["group_size_grpo"]]
                mask[i, prompt_len:] = 1

            loss = compute_grpo_loss(
                new_log_probs,
                old_log_probs_tensor,
                advantages,
                mask,
                config["epsilon_clip"],
                config["beta_kl"]
            )
            loss.backward()

            if (step + 1) % config["gradient_accumulation_steps"] == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["max_grad_norm"])
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1

                if global_rank == 0:
                    wandb.log({
                        "train/loss": loss.item(),
                        "train/learning_rate": optimizer.param_groups[0]["lr"],
                        "train/global_step": global_step
                    })

            if global_step % config["save_steps"] == 0 and global_rank == 0:
                checkpoint_dir = Path(config["output_dir"]) / f"checkpoint-{global_step}"
                model.module.save_pretrained(checkpoint_dir)

    if global_rank == 0:
        model.module.save_pretrained(config["output_dir"])
        wandb.finish()

    cleanup_ddp()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python train.py <path_to_config.yaml>")
        sys.exit(1)
    main(sys.argv[1])