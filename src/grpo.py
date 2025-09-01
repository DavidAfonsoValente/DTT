from accelerate import Accelerator
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import torch
from src.rewards import compute_stage1_reward, compute_stage2_reward
from src.utils import validate_grpo, should_transition
import wandb
import math
from typing import Optional
import torch.nn.functional as F
import os
import sys

torch._dynamo.config.suppress_errors = True


def get_beta_kl(stage: int) -> float:
    return 0.02 if stage == 1 else 0.01


def update_temperature(model, step: int, stage: int, transition_step: int = 0) -> float:
    if stage == 1:
        tau = max(0.8, 2.0 * (0.995 ** (step / 50)))
    else:
        tau = max(0.1, 0.8 * (0.99 ** ((step - transition_step) / 100)))
    model.set_temperature(tau)
    return tau


@torch.no_grad()
def _safe_ref_forward(ref_model, **kwargs):
    return ref_model(**kwargs)


def compute_sequence_logprobs_and_kl(
    model,
    ref_model,
    prompt_ids: torch.Tensor,
    gen_ids_without_prompt: torch.Tensor,
    gen_gates_without_prompt: torch.Tensor,
    training: bool,
    attention_mask: Optional[torch.Tensor] = None,
):
    """
    Computes per-step logprobs for model and ref_model, and an average KL term.
    Assumes single-sequence progression (uses the last hidden state from model forward).
    """
    batch_size = prompt_ids.size(0)
    device = model.device
    prompt_len = prompt_ids.size(1)

    if attention_mask is None:
        attention_mask = torch.ones(batch_size, prompt_len, dtype=torch.long, device=device)
    else:
        attention_mask = attention_mask.to(device)

    # Initial forward pass on the prompt to build caches
    outputs = model(input_ids=prompt_ids.to(device), attention_mask=attention_mask, use_cache=True, output_hidden_states=True)
    past_key_values = outputs.past_key_values

    outputs_ref = _safe_ref_forward(ref_model, input_ids=prompt_ids.to(device), attention_mask=attention_mask, use_cache=True, output_hidden_states=False)
    past_key_values_ref = outputs_ref.past_key_values

    if gen_ids_without_prompt.numel() == 0:
        empty = torch.tensor([], device=device, dtype=torch.float32)
        return empty, empty, torch.tensor(0.0, device=device, dtype=torch.float32)

    logprobs = []
    ref_logprobs = []
    kl_terms = []

    position_id = prompt_len

    # Iterate generated ids
    for t in range(len(gen_ids_without_prompt)):
        next_id = gen_ids_without_prompt[t]
        g = gen_gates_without_prompt[t]

        # Blend embed: gate*hidden + (1-g)*token_embed(next_id)
        # Ensure stable dtype: compute blend in fp32 then cast
        last_hidden = outputs.hidden_states[-1][:, -1, :].to(torch.float32)
        token_embed = model.transformer.wte(next_id.to(device)).to(torch.float32)
        g_val = g if g.dtype == torch.float32 else g.to(torch.float32)
        e = (g_val.unsqueeze(-1) * last_hidden).unsqueeze(1) + ((1.0 - g_val).unsqueeze(-1) * token_embed).unsqueeze(1)
        e = e.to(model.dtype)

        # Extend attention mask
        attention_mask = torch.cat([attention_mask, torch.ones(batch_size, 1, dtype=torch.long, device=device)], dim=1)
        current_position_id = torch.tensor([[position_id]], device=device, dtype=torch.long)

        # Policy forward
        outputs = model(
            inputs_embeds=e,
            position_ids=current_position_id,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
            output_hidden_states=True,
        )
        past_key_values = outputs.past_key_values
        log_soft = F.log_softmax(outputs.logits[:, -1, :], dim=-1)
        logp = log_soft[0, next_id]
        logprobs.append(logp)

        # Reference forward (no grad)
        outputs_ref = _safe_ref_forward(
            ref_model,
            inputs_embeds=e,
            position_ids=current_position_id,
            attention_mask=attention_mask,
            past_key_values=past_key_values_ref,
            use_cache=True,
        )
        past_key_values_ref = outputs_ref.past_key_values
        log_soft_ref = F.log_softmax(outputs_ref.logits[:, -1, :], dim=-1)
        ref_logp = log_soft_ref[0, next_id]
        ref_logprobs.append(ref_logp)

        # KL(log π_ref || log π) with log_target=True is reverse-KL; keep consistent usage
        kl = F.kl_div(log_soft_ref, log_soft, reduction="batchmean", log_target=True)
        kl_terms.append(kl)

        position_id += 1

    logprobs = torch.stack(logprobs)
    ref_logprobs = torch.stack(ref_logprobs)
    avg_kl = torch.mean(torch.stack(kl_terms)) if len(kl_terms) > 0 else torch.tensor(0.0, device=device, dtype=torch.float32)

    return logprobs, ref_logprobs, avg_kl


def train_grpo(model, ref_model, dataset, config, accelerator: Accelerator, collate_fn, tokenizer, debug: bool = False):
    # Data & optim
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True, collate_fn=collate_fn)
    optimizer = AdamW(model.parameters(), lr=config["lr"])

    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    ref_model = accelerator.prepare(ref_model)
    ref_model.eval()

    # Optional model debug switch
    if debug:
        accelerator.unwrap_model(model).debug = True

    # Stage logging
    if accelerator.is_local_main_process:
        wandb.log({"stage": 1, "epoch": 0})

    # Progress bar setup (stdout line buffering and tqdm config)
    if accelerator.is_local_main_process:
        try:
            sys.stdout.reconfigure(line_buffering=True)  # Python 3.7+
        except Exception:
            pass

    total_steps = config["epochs"] * len(dataloader)
    pbar = None
    if accelerator.is_local_main_process:
        pbar = tqdm(
            total=total_steps,
            desc="Training",
            disable=False,
            file=sys.stdout,  # switch to sys.stderr if stdout is buffered by the launcher
            mininterval=0.0,
            miniters=1,
            smoothing=0,
            dynamic_ncols=True,
            leave=True,
        )

    # Training state
    step = 0
    stage = 1
    transition_step = 0
    transition_history = {"structure_rate": [], "gate_ratio": [], "basic_accuracy": []}

    for epoch in range(config["epochs"]):
        model.train()

        epoch_reward_total = 0.0
        epoch_struct = 0.0
        epoch_gate = 0.0
        epoch_basic = 0.0
        epoch_corr = 0.0
        epoch_eff = 0.0
        epoch_loss = 0.0
        epoch_kl = 0.0
        num_samples = 0

        for batch in dataloader:
            batch_size = batch["input_ids"].size(0)
            completions = []
            gates_list = []
            rewards = []
            reward_dicts = []
            advantages = []
            prompts = []

            # Group sampling
            for prompt_idx in range(batch_size):
                prompt_mask_row = batch["attention_mask"][prompt_idx]
                effective_len = prompt_mask_row.sum().item()
                if effective_len == 0:
                    if debug and accelerator.is_local_main_process:
                        print(f"[DEBUG] Skipping empty prompt at batch index {prompt_idx}", flush=True)
                    continue

                prompt_ids = batch["input_ids"][prompt_idx : prompt_idx + 1]
                answer_gt = batch["answer_gt"][prompt_idx]
                group_completions = []
                group_gates = []
                group_rewards = []

                for _ in range(config["group_size"]):
                    unwrapped_model = accelerator.unwrap_model(model)
                    with torch.no_grad():
                        gen_ids, gen_gates = unwrapped_model.generate(
                            prompt_ids,
                            max_length=config["max_length"],
                            do_sample=True,
                            top_p=0.9,
                            temperature=0.8,
                            return_gates=True,
                            training=True,
                            attention_mask=batch["attention_mask"][prompt_idx : prompt_idx + 1],
                        )

                    gen_ids_without_prompt = gen_ids[0, effective_len:]
                    gen_gates_without_prompt = gen_gates[0, :]

                    if stage == 1:
                        reward_dict = compute_stage1_reward(
                            gen_ids_without_prompt,
                            gen_gates_without_prompt,
                            tokenizer,
                            answer_gt,
                            unwrapped_model.bot_id,
                            unwrapped_model.eot_id,
                            config["dataset"],
                        )
                    else:
                        reward_dict = compute_stage2_reward(
                            gen_ids_without_prompt,
                            gen_gates_without_prompt,
                            tokenizer,
                            answer_gt,
                            unwrapped_model.bot_id,
                            unwrapped_model.eot_id,
                            config["dataset"],
                        )

                    group_completions.append(gen_ids_without_prompt)
                    group_gates.append(gen_gates_without_prompt)
                    group_rewards.append(reward_dict["total"])
                    reward_dicts.append(reward_dict)

                # Group-normalized advantage
                mu = sum(group_rewards) / max(1, len(group_rewards))
                var = sum((r - mu) ** 2 for r in group_rewards) / max(1, len(group_rewards))
                sigma = math.sqrt(var + 1e-8)
                group_advantages = [(r - mu) / (sigma if sigma > 0 else 1.0) for r in group_rewards]

                completions.extend(group_completions)
                gates_list.extend(group_gates)
                advantages.extend(group_advantages)
                rewards.extend(group_rewards)
                prompts.extend([prompt_ids] * len(group_completions))

            # Aggregate stats
            epoch_reward_total += sum(rewards)
            epoch_struct += sum(d["struct"] for d in reward_dicts)
            epoch_gate += sum(d["gate"] for d in reward_dicts)
            if stage == 1:
                epoch_basic += sum(d["basic"] for d in reward_dicts)
            else:
                epoch_corr += sum(d["corr"] for d in reward_dicts)
                epoch_eff += sum(d["eff"] for d in reward_dicts)
            num_samples += len(rewards)

            # Policy updates (μ PPO steps)
            for _ in range(config["mu"]):
                batch_loss = 0.0
                batch_kl = 0.0

                for i in range(len(completions)):
                    A = advantages[i]
                    prompt = prompts[i]
                    comp = completions[i]
                    gates = gates_list[i]

                    logprobs, ref_logprobs, avg_kl = compute_sequence_logprobs_and_kl(
                        model,
                        ref_model,
                        prompt.unsqueeze(0),
                        comp,
                        gates,
                        model.training,
                        attention_mask=batch["attention_mask"][i : i + 1] if "attention_mask" in batch else None,
                    )

                    if logprobs.numel() == 0:
                        ppo_term = torch.tensor(0.0, device=model.device, dtype=torch.float32)
                    else:
                        # Stabilize numerics
                        logprobs = torch.nan_to_num(logprobs, nan=0.0, neginf=0.0)
                        ref_logprobs = torch.nan_to_num(ref_logprobs, nan=0.0, neginf=0.0)
                        log_ratios = logprobs - ref_logprobs
                        ratios = torch.exp(log_ratios)
                        surr1 = ratios * A
                        surr2 = torch.clamp(ratios, 1 - config["epsilon"], 1 + config["epsilon"]) * A
                        ppo_term = torch.mean(torch.min(surr1, surr2))

                    batch_kl += float(avg_kl.detach())

                    beta_kl = get_beta_kl(stage)
                    loss_i = -ppo_term + beta_kl * avg_kl
                    batch_loss += loss_i

                if len(completions) > 0:
                    batch_loss = batch_loss / len(completions)
                    accelerator.backward(batch_loss)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()

                    epoch_loss += float(batch_loss.detach())
                    epoch_kl += batch_kl / max(1, len(completions))

            # Advance step and update progress bar BEFORE heavy ops
            step += 1
            if accelerator.is_local_main_process and pbar is not None:
                pbar.update(1)
                # Optional: smaller-grain heartbeat during long inner loops
                if step % 10 == 0:
                    pbar.refresh()

            # Temperature schedule
            if step % 50 == 0:
                tau = update_temperature(model, step, stage, transition_step)
                if accelerator.is_local_main_process:
                    wandb.log({"temperature": tau, "step": step})

            # Stage-1 transition monitoring
            if step % 200 == 0 and stage == 1:
                metrics = validate_grpo(model, config, accelerator, tokenizer, stage, debug=debug)
                transition_history["structure_rate"].append(metrics["structure_rate"])
                transition_history["gate_ratio"].append(metrics["gate_ratio"])
                transition_history["basic_accuracy"].append(metrics["basic_accuracy"])

                if should_transition(metrics, transition_history):
                    stage = 2
                    transition_step = step
                    # Refresh reference policy from current model
                    ref_model.load_state_dict(model.state_dict())
                    if accelerator.is_local_main_process:
                        wandb.log({"stage": stage, "transition_step": step})

            # Periodic reference refresh and checkpointing (AFTER bar update)
            if step % 1000 == 0:
                ref_model.load_state_dict(model.state_dict())
                if accelerator.is_main_process:
                    os.makedirs("checkpoints", exist_ok=True)
                    torch.save(accelerator.get_state_dict(model), f"checkpoints/model_step_{step}.pth")

            # Periodic validation logging
            if step % 500 == 0:
                metrics = validate_grpo(model, config, accelerator, tokenizer, stage, debug=debug)
                if accelerator.is_local_main_process:
                    wandb.log({**metrics, "step": step})

        # Epoch-level logging
        denom = max(1, num_samples)
        log_dict = {
            "epoch": epoch,
            "avg_reward": epoch_reward_total / denom,
            "avg_struct": epoch_struct / denom,
            "avg_gate": epoch_gate / denom,
            "avg_loss": epoch_loss / denom,
            "avg_kl": epoch_kl / denom,
        }
        if stage == 1:
            log_dict["avg_basic"] = epoch_basic / denom
        else:
            log_dict["avg_corr"] = epoch_corr / denom
            log_dict["avg_eff"] = epoch_eff / denom

        if accelerator.is_local_main_process:
            wandb.log(log_dict)

    if accelerator.is_local_main_process and pbar is not None:
        pbar.close()