# src/grpo.py
from accelerate import Accelerator
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
from src.rewards import compute_stage1_reward, compute_stage2_reward
from src.utils import validate_grpo, should_transition
import wandb
import math
from typing import Optional
import torch.nn.functional as F
import os

torch._dynamo.config.suppress_errors = True

def get_beta_kl(stage):
    return 0.02 if stage == 1 else 0.01

def update_temperature(model, step, stage, transition_step=0):
    if stage == 1:
        tau = max(0.8, 2.0 * (0.995 ** (step / 50)))
    else:
        tau = max(0.1, 0.8 * (0.99 ** ((step - transition_step) / 100)))
    model.set_temperature(tau)
    return tau

def compute_sequence_logprobs_and_kl(model, ref_model, prompt_ids, gen_ids_without_prompt, gen_gates_without_prompt, training, attention_mask: Optional[torch.Tensor] = None):
    batch_size = prompt_ids.size(0)
    device = model.device
    prompt_len = prompt_ids.size(1)
    if attention_mask is None:
        attention_mask = torch.ones(batch_size, prompt_len, dtype=torch.long, device=device)
    else:
        attention_mask = attention_mask.to(device)
    past_key_values = None
    past_key_values_ref = None
    logprobs = []
    ref_logprobs = []
    kl_terms = []
    position_id = prompt_len

    outputs = model(input_ids=prompt_ids.to(device), attention_mask=attention_mask)
    past_key_values = outputs.past_key_values

    with torch.no_grad():
        outputs_ref = ref_model(input_ids=prompt_ids.to(device), attention_mask=attention_mask)
    past_key_values_ref = outputs_ref.past_key_values

    if len(gen_ids_without_prompt) == 0:
        return torch.tensor([], device=device), torch.tensor([], device=device), torch.tensor(0.0, device=device)

    for t, next_id in enumerate(gen_ids_without_prompt):
        g = gen_gates_without_prompt[t]
        e = g.unsqueeze(-1) * outputs.hidden_states[-1][:, -1, :].unsqueeze(1) + (1 - g).unsqueeze(-1) * model.transformer.wte(next_id.to(device).unsqueeze(0).unsqueeze(0))  # Linear
        attention_mask = torch.cat([attention_mask, torch.ones(batch_size, 1, dtype=torch.long, device=device)], dim=1)
        current_position_id = torch.tensor([[position_id]], device=device)
        outputs = model(inputs_embeds=e, position_ids=current_position_id, attention_mask=attention_mask, past_key_values=past_key_values, use_cache=True)
        past_key_values = outputs.past_key_values
        log_soft = F.log_softmax(outputs.logits[:, -1, :], dim=-1)
        logp = log_soft[0, next_id]
        logprobs.append(logp)

        with torch.no_grad():
            outputs_ref = ref_model(inputs_embeds=e, position_ids=current_position_id, attention_mask=attention_mask, past_key_values=past_key_values_ref, use_cache=True)
        past_key_values_ref = outputs_ref.past_key_values
        log_soft_ref = F.log_softmax(outputs_ref.logits[:, -1, :], dim=-1)
        ref_logp = log_soft_ref[0, next_id]
        ref_logprobs.append(ref_logp)
        kl = F.kl_div(log_soft_ref, log_soft, reduction='batchmean', log_target=True)
        kl_terms.append(kl)

        position_id += 1

    logprobs = torch.stack(logprobs)
    ref_logprobs = torch.stack(ref_logprobs)
    avg_kl = torch.mean(torch.stack(kl_terms))
    return logprobs, ref_logprobs, avg_kl

def train_grpo(model, ref_model, dataset, config, accelerator, collate_fn, tokenizer, debug=False):
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, collate_fn=collate_fn)
    optimizer = AdamW(model.parameters(), lr=config['lr'])

    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    ref_model = accelerator.prepare(ref_model)
    ref_model.eval()

    if debug:
        accelerator.unwrap_model(model).debug = True

    if accelerator.is_local_main_process:
        wandb.log({"stage": 1, "epoch": 0})

    step = 0
    stage = 1
    transition_step = 0
    transition_history = {'structure_rate': [], 'gate_ratio': [], 'basic_accuracy': []}

    max_steps = config['epochs'] * len(dataloader)
    pbar = tqdm(total=max_steps, desc="Training", disable=not accelerator.is_local_main_process)

    for epoch in range(config['epochs']):
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
            batch_size = batch['input_ids'].size(0)
            completions = []
            gates_list = []
            rewards = []
            reward_dicts = []
            advantages = []
            prompts = []

            for prompt_idx in range(batch_size):
                prompt_mask_row = batch['attention_mask'][prompt_idx]
                effective_len = prompt_mask_row.sum().item()
                if effective_len == 0:
                    if debug:
                        print(f"[DEBUG] Skipping empty prompt at batch index {prompt_idx}")
                    continue
                prompt_ids = batch['input_ids'][prompt_idx : prompt_idx + 1]
                answer_gt = batch['answer_gt'][prompt_idx]
                group_completions = []
                group_gates = []
                group_rewards = []

                for gen_idx in range(config['group_size']):
                    unwrapped_model = accelerator.unwrap_model(model)
                    with torch.no_grad():
                        gen_ids, gen_gates = unwrapped_model.generate(
                            prompt_ids, max_length=config['max_length'], do_sample=True, top_p=0.9, temperature=0.8, return_gates=True, training=True,
                            attention_mask=batch['attention_mask'][prompt_idx : prompt_idx + 1]
                        )
                    gen_ids_without_prompt = gen_ids[0, effective_len:]
                    gen_gates_without_prompt = gen_gates[0, :]

                    if stage == 1:
                        reward_dict = compute_stage1_reward(
                            gen_ids_without_prompt, gen_gates_without_prompt, tokenizer, answer_gt, unwrapped_model.bot_id, unwrapped_model.eot_id, config['dataset']
                        )
                    else:
                        reward_dict = compute_stage2_reward(
                            gen_ids_without_prompt, gen_gates_without_prompt, tokenizer, answer_gt, unwrapped_model.bot_id, unwrapped_model.eot_id, config['dataset']
                        )
                    group_completions.append(gen_ids_without_prompt)
                    group_gates.append(gen_gates_without_prompt)
                    group_rewards.append(reward_dict['total'])
                    reward_dicts.append(reward_dict)

                mu = sum(group_rewards) / config['group_size']
                sigma = math.sqrt(sum((r - mu) ** 2 for r in group_rewards) / config['group_size'] + 1e-8)
                group_advantages = [(r - mu) / sigma for r in group_rewards]

                completions.extend(group_completions)
                gates_list.extend(group_gates)
                advantages.extend(group_advantages)
                rewards.extend(group_rewards)
                prompts.extend([prompt_ids[0]] * config['group_size'])

            epoch_reward_total += sum(rewards)
            epoch_struct += sum(d['struct'] for d in reward_dicts)
            epoch_gate += sum(d['gate'] for d in reward_dicts)
            if stage == 1:
                epoch_basic += sum(d['basic'] for d in reward_dicts)
            else:
                epoch_corr += sum(d['corr'] for d in reward_dicts)
                epoch_eff += sum(d['eff'] for d in reward_dicts)
            num_samples += len(rewards)

            for _ in range(config['mu']):
                batch_loss = 0.0
                batch_kl = 0.0
                for i in range(len(completions)):
                    A = advantages[i]
                    prompt = prompts[i]
                    comp = completions[i]
                    gates = gates_list[i]

                    logprobs, ref_logprobs, avg_kl = compute_sequence_logprobs_and_kl(
                        model, ref_model, prompt.unsqueeze(0), comp, gates, model.training,
                        attention_mask=batch['attention_mask'][i : i + 1]
                    )

                    if logprobs.numel() == 0:
                        ppo_term = torch.tensor(0.0, device=model.device)
                    else:
                        logprobs = torch.nan_to_num(logprobs, nan=0.0, neginf=0.0)
                        ref_logprobs = torch.nan_to_num(ref_logprobs, nan=0.0, neginf=0.0)
                        log_ratios = logprobs - ref_logprobs
                        ratios = torch.exp(log_ratios)
                        surr1 = ratios * A
                        surr2 = torch.clamp(ratios, 1 - config['epsilon'], 1 + config['epsilon']) * A
                        ppo_term = torch.mean(torch.min(surr1, surr2))

                    batch_kl += avg_kl.item()

                    beta_kl = get_beta_kl(stage)
                    loss_i = -ppo_term + beta_kl * avg_kl
                    batch_loss += loss_i

                if len(completions) > 0:
                    batch_loss /= len(completions)
                    accelerator.backward(batch_loss)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()

                    epoch_loss += batch_loss.item()
                    epoch_kl += batch_kl / len(completions)

            step += 1
            pbar.update(1)

            if step % 50 == 0:
                tau = update_temperature(model, step, stage, transition_step)
                if accelerator.is_local_main_process:
                    wandb.log({"temperature": tau, "step": step})

            if step % 200 == 0 and stage == 1:
                metrics = validate_grpo(model, config, accelerator, tokenizer, stage, debug=debug)
                transition_history['structure_rate'].append(metrics['structure_rate'])
                transition_history['gate_ratio'].append(metrics['gate_ratio'])
                transition_history['basic_accuracy'].append(metrics['basic_accuracy'])

                if should_transition(metrics, transition_history):
                    stage = 2
                    transition_step = step
                    ref_model.load_state_dict(model.state_dict())
                    if accelerator.is_local_main_process:
                        wandb.log({"stage": stage, "transition_step": step})

            if step % 1000 == 0:
                ref_model.load_state_dict(model.state_dict())
                if accelerator.is_main_process:
                    torch.save(model.state_dict(), f"checkpoints/model_step_{step}.pth")

            if step % 500 == 0:
                metrics = validate_grpo(model, config, accelerator, tokenizer, stage, debug=debug)
                if accelerator.is_local_main_process:
                    wandb.log(metrics | {"step": step})

        avg_reward = epoch_reward_total / num_samples if num_samples > 0 else 0.0
        avg_struct = epoch_struct / num_samples if num_samples > 0 else 0.0
        avg_gate = epoch_gate / num_samples if num_samples > 0 else 0.0
        avg_loss = epoch_loss / num_samples if num_samples > 0 else 0.0
        avg_kl = epoch_kl / num_samples if num_samples > 0 else 0.0
        log_dict = {
            "epoch": epoch,
            "avg_reward": avg_reward,
            "avg_struct": avg_struct,
            "avg_gate": avg_gate,
            "avg_loss": avg_loss,
            "avg_kl": avg_kl
        }
        if stage == 1:
            log_dict["avg_basic"] = epoch_basic / num_samples if num_samples > 0 else 0.0
        else:
            log_dict["avg_corr"] = epoch_corr / num_samples if num_samples > 0 else 0.0
            log_dict["avg_eff"] = epoch_eff / num_samples if num_samples > 0 else 0.0
        if accelerator.is_local_main_process:
            wandb.log(log_dict)

    pbar.close()