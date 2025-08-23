from torch.utils.data import DataLoader
import torch
from src.rewards import compute_reward
from src.datasets import DTTDataset, collate_fn
import time

def validate_bootstrap(model, config, accelerator, tokenizer, debug=False):
    val_dataset = DTTDataset(config['dataset'], tokenizer, split='valid', synthetic_ratio=0, data_dir=config.get('data_dir', 'data'))
    val_collate = lambda batch: collate_fn(batch, tokenizer.pad_token_id)
    batch_size = config.get('val_batch_size', 8)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=val_collate)
    val_loader = accelerator.prepare(val_loader)
    structure_count = 0
    inner_gates_sum = 0.0
    num_samples = 0

    model.eval()
    with torch.no_grad():
        if accelerator.is_local_main_process:
            for batch_idx, batch in enumerate(val_loader):
                if num_samples >= config['val_size']:
                    break
                if debug:
                    for i in range(batch['input_ids'].size(0)):
                        input_text = tokenizer.decode(batch['input_ids'][i], skip_special_tokens=False)
                        print(f"Decoded input text [{i}]: {input_text[:512]}...")
                
                prompt_outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
                
                gen_start = time.time()
                gen_ids, gates = [], []
                for i in range(batch['input_ids'].size(0)):
                    single_input = batch['input_ids'][i:i+1]
                    single_gen_ids, single_gates = model.generate(single_input, max_length=config['max_length'], return_gates=True)
                    gen_ids.append(single_gen_ids)
                    gates.append(single_gates)
                gen_ids = torch.cat(gen_ids, dim=0)
                gates = torch.cat(gates, dim=0)
                
                for i in range(gen_ids.size(0)):
                    bot_pos = (gen_ids[i] == model.bot_id).nonzero(as_tuple=True)[0]
                    eot_pos = (gen_ids[i] == model.eot_id).nonzero(as_tuple=True)[0]
                    if len(bot_pos) > 0 and len(eot_pos) > 0 and bot_pos[0] < eot_pos[0]:
                        structure_count += 1
                        inner_gates = gates[i, bot_pos[0]+1:eot_pos[0]]
                        mean_inner = inner_gates.mean().item() if inner_gates.numel() > 0 else 0.0
                        inner_gates_sum += mean_inner
                    num_samples += 1
                    if num_samples >= config['val_size']:
                        break

    structure_rate = structure_count / num_samples if num_samples > 0 else 0
    mean_inner_gate = inner_gates_sum / (structure_count or 1)
    
    results = {
        'is_valid': structure_rate >= 0.40 and mean_inner_gate >= 0.60,
        'structure_rate': structure_rate,
        'mean_inner_gate': mean_inner_gate
    }
    return accelerator.gather(results) if accelerator.num_processes > 1 else results

def validate_grpo(model, config, accelerator, tokenizer, debug=False):
    val_dataset = DTTDataset(config['dataset'], tokenizer, split='valid', synthetic_ratio=0, data_dir=config.get('data_dir', 'data'))
    val_collate = lambda batch: collate_fn(batch, tokenizer.pad_token_id)
    batch_size = config.get('val_batch_size', 8)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=val_collate)
    val_loader = accelerator.prepare(val_loader)
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
        if accelerator.is_local_main_process:
            for batch_idx, batch in enumerate(val_loader):
                if num_samples >= 256:
                    break
                if debug:
                    for i in range(batch['input_ids'].size(0)):
                        input_text = tokenizer.decode(batch['input_ids'][i], skip_special_tokens=False)
                        print(f"Decoded input text [{i}]: {input_text[:512]}...")
                
                gen_start = time.time()
                gen_ids, gates = [], []
                for i in range(batch['input_ids'].size(0)):
                    single_input = batch['input_ids'][i:i+1]
                    single_gen_ids, single_gates = model.generate(single_input, max_length=config['max_length'], return_gates=True)
                    gen_ids.append(single_gen_ids)
                    gates.append(single_gates)
                gen_ids = torch.cat(gen_ids, dim=0)
                gates = torch.cat(gates, dim=0)
                
                for i in range(gen_ids.size(0)):
                    reward_dict = compute_reward(gen_ids[i], gates[i], tokenizer, batch['answer_gt'][i], model.bot_id, model.eot_id, config['dataset'], model.dummy_id)
                    total_reward += reward_dict['total']
                    r_struct_sum += reward_dict['struct']
                    r_corr_sum += reward_dict['corr']
                    r_eff_sum += reward_dict['eff']
                    r_gate_sum += reward_dict['gate']
                    bot_pos = (gen_ids[i] == model.bot_id).nonzero(as_tuple=True)[0]
                    eot_pos = (gen_ids[i] == model.eot_id).nonzero(as_tuple=True)[0]
                    has_span = len(bot_pos) > 0 and len(eot_pos) > 0 and bot_pos[0] < eot_pos[0]
                    if has_span:
                        structure_count += 1
                        inner_gates = gates[i, bot_pos[0]+1:eot_pos[0]]
                        mean_inner = inner_gates.mean().item() if inner_gates.numel() > 0 else 0.0
                        inner_gates_sum += mean_inner
                    if reward_dict['corr'] > 0:
                        correct_count += 1
                    num_samples += 1
                    if num_samples >= 256:
                        break

    avg_reward = total_reward / num_samples if num_samples > 0 else 0.0
    avg_r_struct = r_struct_sum / num_samples if num_samples > 0 else 0.0
    avg_r_corr = r_corr_sum / num_samples if num_samples > 0 else 0.0
    avg_r_eff = r_eff_sum / num_samples if num_samples > 0 else 0.0
    avg_r_gate = r_gate_sum / num_samples if num_samples > 0 else 0.0
    structure_rate = structure_count / num_samples if num_samples > 0 else 0.0
    correct_rate = correct_count / num_samples if num_samples > 0 else 0.0
    mean_inner_gate = inner_gates_sum / structure_count if structure_count > 0 else 0.0

    results = {
        'avg_reward': avg_reward,
        'avg_r_struct': avg_r_struct,
        'avg_r_corr': avg_r_corr,
        'avg_r_eff': avg_r_eff,
        'avg_r_gate': avg_r_gate,
        'structure_rate': structure_rate,
        'correct_rate': correct_rate,
        'mean_inner_gate': mean_inner_gate
    }
    return accelerator.gather(results) if accelerator.num_processes > 1 else results