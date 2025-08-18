from torch.utils.data import DataLoader
import torch
from rewards import compute_reward
from src.datasets import DTTDataset

def validate_bootstrap(model, config, accelerator, tokenizer):
    val_dataset = DTTDataset(config['dataset'], tokenizer, split='valid', data_dir=config.get('data_dir', 'data'))
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
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
                inner_gates_sum += inner_gates.mean().item()
            num_samples += 1
    
    structure_rate = structure_count / num_samples if num_samples > 0 else 0
    mean_inner_gate = inner_gates_sum / (structure_count or 1)
    return {'is_valid': structure_rate >= 0.40 and mean_inner_gate >= 0.60, 'structure_rate': structure_rate, 'mean_inner_gate': mean_inner_gate}

def validate_grpo(model, config, accelerator, tokenizer):
    val_dataset = DTTDataset(config['dataset'], tokenizer, split='valid', data_dir=config.get('data_dir', 'data'))
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    total_reward = 0.0
    num_samples = 0
    
    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            if num_samples >= 256: break
            gen_ids, gates = model.generate(batch['input_ids'], max_length=config['max_length'], return_gates=True)
            reward = compute_reward(gen_ids[0], gates[0], tokenizer, batch['answer_gt'][0], model.bot_id, model.eot_id)
            total_reward += reward
            num_samples += 1
    
    return total_reward / num_samples if num_samples > 0 else 0.0