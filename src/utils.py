import torch
from tqdm import tqdm
import time
from src.rewards import compute_reward

def validate_bootstrap(model, config, accelerator, tokenizer, debug=False):
    dataset = config['val_dataset']
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=config['collate_fn'])
    model, dataloader = accelerator.prepare(model, dataloader)
    
    if debug:
        accelerator.unwrap_model(model).debug = True
    
    num_samples = 0
    num_structured = 0
    inner_gates = []
    model.eval()
    
    for batch in tqdm(dataloader, desc="Validating bootstrap"):
        if debug and accelerator.is_local_main_process:
            val_start = time.time()
            print(f"Validating sample {num_samples + 1}/{len(dataloader)}")
            input_text = tokenizer.decode(batch['input_ids'][0], skip_special_tokens=False)
            print(f"Input text: {input_text[:512] if len(input_text) > 512 else input_text}")
        
        with torch.no_grad():
            gen_ids, gates = model.generate(batch['input_ids'], max_length=config['max_length'], return_gates=True)
        
        if debug and accelerator.is_local_main_process:
            gen_text = tokenizer.decode(gen_ids[0], skip_special_tokens=False)
            print(f"Validation generate took {time.time() - val_start:.2f}s")
            print(f"Generated text: {gen_text[:512] if len(gen_text) > 512 else gen_text}")
        
        bot_pos = (gen_ids == model.bot_id).nonzero(as_tuple=True)[1]
        eot_pos = (gen_ids == model.eot_id).nonzero(as_tuple=True)[1]
        
        if debug and accelerator.is_local_main_process:
            print(f"bot_id positions: {bot_pos.tolist()}")
            print(f"eot_id positions: {eot_pos.tolist()}")
        
        has_structure = False
        if bot_pos.numel() > 0 and eot_pos.numel() > 0:
            bot_idx = bot_pos[0].item()
            eot_idx = eot_pos[0].item()
            if bot_idx < eot_idx:
                num_structured += 1
                has_structure = True
                inner_gates.append(gates[0, bot_idx + 1:eot_idx].mean().item() if eot_idx > bot_idx + 1 else 0.0)
        
        if debug and accelerator.is_local_main_process:
            if has_structure:
                print(f"Structure detected: [bot] at {bot_idx}, [eot] at {eot_idx}")
            else:
                print("No structure detected")
        
        num_samples += 1
        if num_samples >= config['val_size']:
            break
    
    structure_rate = num_structured / num_samples if num_samples > 0 else 0.0
    mean_inner_gate = sum(inner_gates) / len(inner_gates) if inner_gates else 0.0
    
    if debug and accelerator.is_local_main_process:
        print(f"Validation complete: Structure rate = {structure_rate:.4f}, Mean inner gate = {mean_inner_gate:.4f}")
    
    return {
        'structure_rate': structure_rate,
        'mean_inner_gate': mean_inner_gate,
        'is_valid': structure_rate >= 0.40 and mean_inner_gate >= 0.60
    }

def validate_grpo(model, config, accelerator, tokenizer, debug=False):
    dataset = config['val_dataset']
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=config['collate_fn'])
    model, dataloader = accelerator.prepare(model, dataloader)
    
    if debug:
        accelerator.unwrap_model(model).debug = True
    
    num_samples = 0
    num_structured = 0
    num_correct = 0
    inner_gates = []
    rewards = []
    r_struct = []
    r_corr = []
    r_eff = []
    r_gate = []
    model.eval()
    
    for batch in tqdm(dataloader, desc="Validating GRPO"):
        if debug and accelerator.is_local_main_process:
            val_start = time.time()
            print(f"Validating sample {num_samples + 1}/{len(dataloader)}")
            input_text = tokenizer.decode(batch['input_ids'][0], skip_special_tokens=False)
            print(f"Input text: {input_text[:512] if len(input_text) > 512 else input_text}")
            print(f"Ground truth answer: {batch['answer_gt'][0]}")
        
        with torch.no_grad():
            gen_ids, gates = model.generate(batch['input_ids'], max_length=config['max_length'], return_gates=True)
        
        if debug and accelerator.is_local_main_process:
            gen_text = tokenizer.decode(gen_ids[0], skip_special_tokens=False)
            print(f"Validation generate took {time.time() - val_start:.2f}s")
            print(f"Generated text: {gen_text[:512] if len(gen_text) > 512 else gen_text}")
        
        bot_pos = (gen_ids == model.bot_id).nonzero(as_tuple=True)[1]
        eot_pos = (gen_ids == model.eot_id).nonzero(as_tuple=True)[1]
        
        if debug and accelerator.is_local_main_process:
            print(f"bot_id positions: {bot_pos.tolist()}")
            print(f"eot_id positions: {eot_pos.tolist()}")
        
        has_structure = False
        if bot_pos.numel() > 0 and eot_pos.numel() > 0:
            bot_idx = bot_pos[0].item()
            eot_idx = eot_pos[0].item()
            if bot_idx < eot_idx:
                num_structured += 1
                has_structure = True
                inner_gates.append(gates[0, bot_idx + 1:eot_idx].mean().item() if eot_idx > bot_idx + 1 else 0.0)
        
        if debug and accelerator.is_local_main_process:
            if has_structure:
                print(f"Structure detected: [bot] at {bot_idx}, [eot] at {eot_idx}")
            else:
                print("No structure detected")
        
        reward_dict = compute_reward(
            gen_ids[0], gates[0], tokenizer, batch['answer_gt'][0],
            model.bot_id, model.eot_id, config['dataset'], model.dummy_id,
            weights={'struct': 0.6, 'corr': 1.0, 'eff': 0.25, 'gate': 0.6}
        )
        
        if debug and accelerator.is_local_main_process:
            print(f"Reward dict: {reward_dict}")
        
        rewards.append(reward_dict['total'])
        r_struct.append(reward_dict['struct'])
        r_corr.append(reward_dict['corr'])
        r_eff.append(reward_dict['eff'])
        r_gate.append(reward_dict['gate'])
        if reward_dict['corr'] > 0.5:
            num_correct += 1
        
        num_samples += 1
        if num_samples >= config['val_size']:
            break
    
    avg_reward = sum(rewards) / num_samples if num_samples > 0 else 0.0
    structure_rate = num_structured / num_samples if num_samples > 0 else 0.0
    correct_rate = num_correct / num_samples if num_samples > 0 else 0.0
    mean_inner_gate = sum(inner_gates) / len(inner_gates) if inner_gates else 0.0
    avg_r_struct = sum(r_struct) / num_samples if num_samples > 0 else 0.0
    avg_r_corr = sum(r_corr) / num_samples if num_samples > 0 else 0.0
    avg_r_eff = sum(r_eff) / num_samples if num_samples > 0 else 0.0
    avg_r_gate = sum(r_gate) / num_samples if num_samples > 0 else 0.0
    
    if debug and accelerator.is_local_main_process:
        print(f"Validation complete: Structure rate = {structure_rate:.4f}, Correct rate = {correct_rate:.4f}, "
              f"Mean inner gate = {mean_inner_gate:.4f}, Avg reward = {avg_reward:.4f}")
    
    return {
        'avg_reward': avg_reward,
        'structure_rate': structure_rate,
        'correct_rate': correct_rate,
        'mean_inner_gate': mean_inner_gate,
        'avg_r_struct': avg_r_struct,
        'avg_r_corr': avg_r_corr,
        'avg_r_eff': avg_r_eff,
        'avg_r_gate': avg_r_gate
    }