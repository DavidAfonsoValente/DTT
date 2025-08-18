from accelerate import Accelerator
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
from src.rewards import compute_reward
from src.utils import validate_grpo
import wandb
from src.model import DTTModel
import math
import time

def smooth(k, a, b):
    mid = (a + b) / 2
    scale = (b - a) / 6
    return 1 / (1 + math.exp(- (k - mid) / scale))

def get_weights(step):
    if step < 1500:
        return {'struct': 1.0, 'corr': 0.4, 'eff': 0.0, 'gate': 0.6}
    elif step < 4000:
        s = smooth(step, 1500, 3500)
        w_struct = 1.0 - 0.3 * s
        w_corr = 0.4 + 0.6 * s
        w_eff = 0.0 + 0.15 * s
        w_gate = 0.6 - 0.1 * s
        return {'struct': w_struct, 'corr': w_corr, 'eff': w_eff, 'gate': w_gate}
    elif step < 7000:
        s = smooth(step, 4000, 6500)
        w_struct = 0.7 - 0.1 * s
        w_corr = 1.0
        w_eff = 0.15 + 0.15 * s
        w_gate = 0.5 + 0.1 * s
        return {'struct': w_struct, 'corr': w_corr, 'eff': w_eff, 'gate': w_gate}
    else:
        return {'struct': 0.6, 'corr': 1.0, 'eff': 0.3, 'gate': 0.6}

def train_grpo(model, dataset, config, accelerator, ref_checkpoint, collate_fn, debug=False):
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, collate_fn=collate_fn)
    optimizer = AdamW(model.parameters(), lr=config['lr'])
    
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    ref_model = DTTModel.from_pretrained(ref_checkpoint)
    ref_model = accelerator.prepare(ref_model)
    ref_model.eval()
    
    if accelerator.is_local_main_process:
        wandb.log({"stage": "grpo_start", "epoch": 0})
    
    step = 0
    prev_val_reward = -float('inf')
    plateau_steps = 0
    current_phase = 1
    phase_thresholds = [0, 1500, 4000, 7000]
    phase_conditions = [
        None,
        lambda m: m['structure_rate'] >= 0.7 and m['mean_inner_gate'] >= 0.6,
        lambda m: m['structure_rate'] >= 0.85 and m['correct_rate'] >= 0.4,
        lambda m: m['structure_rate'] >= 0.9 and m['correct_rate'] >= 0.6,
    ]
    next_transition_attempt = 0
    prev_metrics = None
    transition_metrics = None
    transition_step = 0
    
    for epoch in range(config['epochs']):
        if debug and accelerator.is_local_main_process:
            print(f"Starting epoch {epoch + 1}/{config['epochs']}")
            epoch_start = time.time()
        
        model.train()
        epoch_reward = 0.0
        epoch_r_struct = 0.0
        epoch_r_corr = 0.0
        epoch_r_eff = 0.0
        epoch_r_gate = 0.0
        epoch_kl = 0.0
        num_samples = 0
        
        for batch_idx, batch in enumerate(tqdm(dataloader)):
            if debug and accelerator.is_local_main_process:
                batch_start = time.time()
                print(f"Processing batch {batch_idx + 1}/{len(dataloader)}")
                print(f"Batch input_ids shape: {batch['input_ids'].shape}")
                print(f"Batch answer_gt: {batch['answer_gt']}")
            
            batch_loss = 0.0
            for prompt_idx in range(batch['input_ids'].size(0)):
                prompt_ids = batch['input_ids'][prompt_idx:prompt_idx+1]
                answer_gt = batch['answer_gt'][prompt_idx]
                if debug and accelerator.is_local_main_process:
                    print(f"  Prompt {prompt_idx + 1}: prompt_ids shape {prompt_ids.shape}, GT answer: {answer_gt}")
                
                completions = []
                rewards = []
                reward_dicts = []
                weights = get_weights(step)
                for gen_idx in range(config['group_size']):
                    if debug and accelerator.is_local_main_process:
                        gen_start = time.time()
                    gen_ids, gates = model.generate(prompt_ids, max_length=config['max_length'], do_sample=True, top_p=0.95, return_gates=True)
                    if debug and accelerator.is_local_main_process:
                        print(f"    Generation {gen_idx + 1}/{config['group_size']} took {time.time() - gen_start:.2f}s")
                        gen_text = model.tokenizer.decode(gen_ids[0], skip_special_tokens=False)
                        print(f"    Generated text: {gen_text[:100]}...")  # Truncated for brevity
                    
                    reward_dict = compute_reward(gen_ids[0], gates[0], model.tokenizer, answer_gt, model.bot_id, model.eot_id, config['dataset'], model.dummy_id, weights=weights)
                    if debug and accelerator.is_local_main_process:
                        print(f"    Reward dict: {reward_dict}")
                    
                    completions.append(gen_ids)
                    rewards.append(reward_dict['total'])
                    reward_dicts.append(reward_dict)
                
                num_samples += config['group_size']
                epoch_reward += sum(rewards)
                epoch_r_struct += sum(d['struct'] for d in reward_dicts)
                epoch_r_corr += sum(d['corr'] for d in reward_dicts)
                epoch_r_eff += sum(d['eff'] for d in reward_dicts)
                epoch_r_gate += sum(d['gate'] for d in reward_dicts)
                
                mu = sum(rewards) / config['group_size']
                sigma = (sum((r - mu)**2 for r in rewards) / config['group_size'])**0.5 + 1e-8
                advantages = [(r - mu) / sigma for r in rewards]
                if debug and accelerator.is_local_main_process:
                    print(f"  Advantages: {advantages}")
                
                for i, comp_ids in enumerate(completions):
                    with torch.no_grad():
                        outputs_old = ref_model(comp_ids, labels=comp_ids)
                    outputs = model(comp_ids, labels=comp_ids)
                    logprobs = outputs.logits[:, :-1, :].log_softmax(-1).gather(2, comp_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
                    logprobs_old = outputs_old.logits[:, :-1, :].log_softmax(-1).gather(2, comp_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
                    ratios = torch.exp(logprobs - logprobs_old)
                    mean_ratio = ratios.mean()
                    surr1 = mean_ratio * advantages[i]
                    surr2 = torch.clamp(mean_ratio, 1 - config['epsilon'], 1 + config['epsilon']) * advantages[i]
                    ppo_loss = -torch.min(surr1, surr2)
                    
                    kl = torch.nn.functional.kl_div(logprobs_old, logprobs, log_target=True, reduction='mean')
                    loss = ppo_loss + config['beta_kl'] * kl
                    batch_loss += loss
                    
                    epoch_kl += kl.item()
                    if debug and accelerator.is_local_main_process:
                        print(f"    Completion {i + 1}: PPO loss {ppo_loss.item():.4f}, KL {kl.item():.4f}, Total loss {loss.item():.4f}")
            
            batch_loss /= batch['input_ids'].size(0)
            if debug and accelerator.is_local_main_process:
                print(f"  Batch loss: {batch_loss.item():.4f}")
            
            accelerator.backward(batch_loss)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            
            step += 1
            if step % 1000 == 0:
                model.set_temperature(max(0.1, 2.0 * (0.9 ** (step / 1000))))
                val_metrics = validate_grpo(model, config, accelerator, model.tokenizer, debug=debug)
                if accelerator.is_local_main_process:
                    wandb.log({
                        "grpo/step": step,
                        "grpo/validation_reward": val_metrics['avg_reward'],
                        "grpo/validation_structure_rate": val_metrics['structure_rate'],
                        "grpo/validation_correct_rate": val_metrics['correct_rate'],
                        "grpo/validation_mean_inner_gate": val_metrics['mean_inner_gate'],
                        "grpo/validation_avg_r_struct": val_metrics['avg_r_struct'],
                        "grpo/validation_avg_r_corr": val_metrics['avg_r_corr'],
                        "grpo/validation_avg_r_eff": val_metrics['avg_r_eff'],
                        "grpo/validation_avg_r_gate": val_metrics['avg_r_gate'],
                    })
                    if debug:
                        print(f"Validation metrics at step {step}: {val_metrics}")
                if val_metrics['avg_reward'] < prev_val_reward * 1.02:
                    plateau_steps += 1000
                else:
                    plateau_steps = 0
                if plateau_steps >= 5000:
                    model.set_temperature(model.temperature / 2)
                    plateau_steps = 0
                prev_val_reward = val_metrics['avg_reward']

                # Phase transition logic
                prev_phase = current_phase
                if step >= next_transition_attempt and current_phase < 4 and step >= phase_thresholds[current_phase] and phase_conditions[current_phase](val_metrics):
                    current_phase += 1
                    transition_metrics = prev_metrics
                    transition_step = step
                    next_transition_attempt = 0
                    if debug and accelerator.is_local_main_process:
                        print(f"Transitioned to phase {current_phase}")

                # Degradation check
                if prev_phase < current_phase and step - transition_step <= 5000:
                    degrade = False
                    key_metrics = ['structure_rate', 'correct_rate']
                    for km in key_metrics:
                        if val_metrics[km] < transition_metrics[km] * 0.85:
                            degrade = True
                            break
                    if degrade:
                        current_phase -= 1
                        next_transition_attempt = step + 1000
                        if debug and accelerator.is_local_main_process:
                            print(f"Degradation detected, reverted to phase {current_phase}")

                prev_metrics = val_metrics
            
            if debug and accelerator.is_local_main_process:
                print(f"Batch {batch_idx + 1} processed in {time.time() - batch_start:.2f}s")
        
        avg_reward = epoch_reward / num_samples if num_samples > 0 else 0
        avg_r_struct = epoch_r_struct / num_samples if num_samples > 0 else 0
        avg_r_corr = epoch_r_corr / num_samples if num_samples > 0 else 0
        avg_r_eff = epoch_r_eff / num_samples if num_samples > 0 else 0
        avg_r_gate = epoch_r_gate / num_samples if num_samples > 0 else 0
        avg_kl = epoch_kl / num_samples if num_samples > 0 else 0
        
        if accelerator.is_local_main_process:
            wandb.log({
                "grpo/epoch": epoch + 1,
                "grpo/avg_reward": avg_reward,
                "grpo/avg_r_struct": avg_r_struct,
                "grpo/avg_r_corr": avg_r_corr,
                "grpo/avg_r_eff": avg_r_eff,
                "grpo/avg_r_gate": avg_r_gate,
                "grpo/avg_kl": avg_kl
            })
        
        if debug and accelerator.is_local_main_process:
            print(f"Epoch {epoch + 1} completed in {time.time() - epoch_start:.2f}s | Avg Reward: {avg_reward:.4f} | Avg KL: {avg_kl:.4f}")