# src/utils.py
from torch.utils.data import DataLoader
import torch
from src.rewards import compute_reward
from src.datasets import DTTDataset, collate_fn

def validate_bootstrap(model, config, accelerator, tokenizer):
    val_dataset = DTTDataset(config['dataset'], tokenizer, split='valid', synthetic_ratio=0, data_dir=config.get('data_dir', 'data'))
    val_collate = lambda batch: collate_fn(batch, tokenizer.pad_token_id)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=val_collate)
    structure_count = 0
    inner_gates_sum = 0.0
    num_samples = 0
    
    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            if num_samples >= config['val_size']: break
            gen_ids, gates = model.generate(batch['input_ids'], max_length=config['max_length'], return_gates=True)
            bot_pos = (gen_ids == model.bot_id).nonzero(as_tuple=True)[0]
            eot_pos = (gen_ids == model.eot_id).nonzero(as_tuple=True)[0]
            if len(bot_pos) > 0 and len(eot_pos) > 0 and bot_pos[0] < eot_pos[0]:
                structure_count += 1
                inner_gates = gates[:, bot_pos[0]+1:eot_pos[0]]
                inner_gates_sum += inner_gates.mean().item() if inner_gates.numel() > 0 else 0.0
            num_samples += 1
    
    structure_rate = structure_count / num_samples if num_samples > 0 else 0
    mean_inner_gate = inner_gates_sum / (structure_count or 1)
    return {'is_valid': structure_rate >= 0.70 and mean_inner_gate >= 0.60, 'structure_rate': structure_rate, 'mean_inner_gate': mean_inner_gate}

def validate_grpo(model, config, accelerator, tokenizer):
    val_dataset = DTTDataset(config['dataset'], tokenizer, split='valid', synthetic_ratio=0, data_dir=config.get('data_dir', 'data'))
    val_collate = lambda batch: collate_fn(batch, tokenizer.pad_token_id)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=val_collate)
    total_reward = 0.0
    r_struct_sum = 0.0
    r_corr_sum = 0.0
    r_eff_sum = 0.0
    r_gate_sum = 0.0
    structure_count = 0
    correct_count = 0
    inner_gates_sum = 0.0
    num_samples = 0
    
    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            if num_samples >= 256: break
            gen_ids, gates = model.generate(batch['input_ids'], max_length=config['max_length'], return_gates=True)
            reward_dict = compute_reward(gen_ids[0], gates[0], tokenizer, batch['answer_gt'][0], model.bot_id, model.eot_id, config['dataset'], model.dummy_id)
            total_reward += reward_dict['total']
            r_struct_sum += reward_dict['struct']
            r_corr_sum += reward_dict['corr']
            r_eff_sum += reward_dict['eff']
            r_gate_sum += reward_dict['gate']
            bot_pos = (gen_ids[0] == model.bot_id).nonzero(as_tuple=True)[0]
            eot_pos = (gen_ids[0] == model.eot_id).nonzero(as_tuple=True)[0]
            has_span = len(bot_pos) > 0 and len(eot_pos) > 0 and bot_pos[0] < eot_pos[0]
            if has_span:
                structure_count += 1
                inner_gates = gates[0, bot_pos[0]+1:eot_pos[0]]
                inner_gates_sum += inner_gates.mean().item() if inner_gates.numel() > 0 else 0.0
            if reward_dict['corr'] > 0:
                correct_count += 1
            num_samples += 1
    
    avg_reward = total_reward / num_samples if num_samples > 0 else 0.0
    avg_r_struct = r_struct_sum / num_samples if num_samples > 0 else 0.0
    avg_r_corr = r_corr_sum / num_samples if num_samples > 0 else 0.0
    avg_r_eff = r_eff_sum / num_samples if num_samples > 0 else 0.0
    avg_r_gate = r_gate_sum / num_samples if num_samples > 0 else 0.0
    structure_rate = structure_count / num_samples if num_samples > 0 else 0.0
    correct_rate = correct_count / num_samples if num_samples > 0 else 0.0
    mean_inner_gate = inner_gates_sum / structure_count if structure_count > 0 else 0.0
    
    return {
        'avg_reward': avg_reward,
        'avg_r_struct': avg_r_struct,
        'avg_r_corr': avg_r_corr,
        'avg_r_eff': avg_r_eff,
        'avg_r_gate': avg_r_gate,
        'structure_rate': structure_rate,
        'correct_rate': correct_rate,
        'mean_inner_gate': mean_inner_gate
    }