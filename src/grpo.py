# src/grpo.py
from accelerate import Accelerator
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
from src.rewards import compute_reward
from src.utils import validate_grpo
import wandb
from src.datasets import collate_fn

def train_grpo(model, dataset, config, accelerator, ref_checkpoint):
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
    
    for epoch in range(config['epochs']):
        model.train()
        epoch_reward = 0.0
        epoch_r_struct = 0.0
        epoch_r_corr = 0.0
        epoch_r_eff = 0.0
        epoch_r_gate = 0.0
        epoch_kl = 0.0
        num_samples = 0
        
        for batch in tqdm(dataloader):
            batch_loss = 0.0
            for prompt_idx in range(batch['input_ids'].size(0)):
                prompt_ids = batch['input_ids'][prompt_idx:prompt_idx+1]
                answer_gt = batch['answer_gt'][prompt_idx]
                
                completions = []
                rewards = []
                reward_dicts = []
                for _ in range(config['group_size']):
                    gen_ids, gates = model.generate(prompt_ids, max_length=config['max_length'], do_sample=True, top_p=0.95, return_gates=True)
                    reward_dict = compute_reward(gen_ids[0], gates[0], model.tokenizer, answer_gt, model.bot_id, model.eot_id, config['dataset'], model.dummy_id)
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
                    
                    kl = kl_div(logprobs_old, logprobs, log_target=True, reduction='mean')
                    loss = ppo_loss + config['beta_kl'] * kl
                    batch_loss += loss
                    
                    epoch_kl += kl.item()
            
            batch_loss /= batch['input_ids'].size(0)
            accelerator.backward(batch_loss)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            
            step += 1
            if step % 1000 == 0:
                model.set_temperature(max(0.1, 2.0 * (0.9 ** (step / 1000))))
                val_reward = validate_grpo(model, config, accelerator, model.tokenizer)
                if accelerator.is_local_main_process:
                    wandb.log({
                        "grpo/step": step,
                        "grpo/validation_reward": val_reward
                    })
                if val_reward < prev_val_reward * 1.02:
                    plateau_steps += 1000
                else:
                    plateau_steps = 0
                if plateau_steps >= 5000:
                    model.set_temperature(model.temperature / 2)
                    plateau_steps = 0
                prev_val_reward = val_reward
        
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