# src/utils.py
from torch.utils.data import DataLoader
import torch
from src.rewards import compute_stage1_reward, compute_stage2_reward
from src.datasets import DTTDataset, collate_fn
from tqdm import tqdm

def validate_grpo(model, config, accelerator, tokenizer, stage, debug=False):
    val_dataset = DTTDataset(config['dataset'], tokenizer, split='valid', data_dir=config.get('data_dir', 'data'))
    val_loader = DataLoader(val_dataset, batch_size=config.get('valid_batch_size', 8), shuffle=False, collate_fn=lambda batch: collate_fn(batch, tokenizer.pad_token_id))
    val_loader = accelerator.prepare(val_loader)
    model.eval()

    num_samples = 0
    total_reward = 0.0
    num_struct_correct = 0
    total_inner_gate = 0.0
    total_outer_gate = 0.0
    num_spans = 0
    num_correct = 0
    total_think_len = 0.0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation", disable=not accelerator.is_local_main_process):
            batch_size = batch['input_ids'].size(0)
            for prompt_idx in range(batch_size):
                prompt_ids = batch['input_ids'][prompt_idx : prompt_idx + 1]
                if prompt_ids.size(1) == 0:
                    if debug:
                        print(f"[DEBUG] Skipping empty prompt at validation batch index {prompt_idx}")
                    continue
                answer_gt = batch['answer_gt'][prompt_idx]

                gen_ids, gen_gates = model.generate(
                    prompt_ids, max_length=config['max_length'], do_sample=False, return_gates=True, training=False
                )

                gen_ids_without_prompt = gen_ids[0, len(prompt_ids[0]):]
                gen_gates_without_prompt = gen_gates[0, len(prompt_ids[0])-1:]

                if stage == 1:
                    reward_dict = compute_stage1_reward(
                        gen_ids_without_prompt, gen_gates_without_prompt, tokenizer, answer_gt, model.bot_id, model.eot_id, config['dataset']
                    )
                    is_correct = reward_dict['basic'] > 0
                else:
                    reward_dict = compute_stage2_reward(
                        gen_ids_without_prompt, gen_gates_without_prompt, tokenizer, answer_gt, model.bot_id, model.eot_id, config['dataset']
                    )
                    is_correct = reward_dict['corr'] > 1.0  # Exact match

                total_reward += reward_dict['total']
                if reward_dict['struct'] == 2.0:  # Proper [bot]...[eot]
                    num_struct_correct += 1
                    bot_pos = (gen_ids_without_prompt == model.bot_id).nonzero(as_tuple=True)[0][0].item()
                    eot_pos = (gen_ids_without_prompt == model.eot_id).nonzero(as_tuple=True)[0][0].item()
                    inner_gates = gen_gates_without_prompt[bot_pos + 1 : eot_pos]
                    outer_gates = torch.cat([gen_gates_without_prompt[:bot_pos], gen_gates_without_prompt[eot_pos:]])
                    total_inner_gate += inner_gates.mean().item()
                    total_outer_gate += outer_gates.mean().item()
                    think_len = eot_pos - bot_pos - 1
                    total_think_len += think_len
                    num_spans += 1

                if is_correct:
                    num_correct += 1

                num_samples += 1

    structure_rate = num_struct_correct / num_samples if num_samples > 0 else 0.0
    mean_inner_gate = total_inner_gate / num_spans if num_spans > 0 else 0.0
    mean_outer_gate = total_outer_gate / num_spans if num_spans > 0 else 0.0
    gate_ratio = mean_inner_gate / (mean_outer_gate + 0.1) if mean_outer_gate > 0 else 0.0
    accuracy = num_correct / num_samples if num_samples > 0 else 0.0
    avg_think_len = total_think_len / num_spans if num_spans > 0 else 0.0
    avg_reward = total_reward / num_samples if num_samples > 0 else 0.0

    if debug and accelerator.is_local_main_process:
        print(f"[DEBUG] Validation metrics: structure_rate={structure_rate:.3f}, gate_ratio={gate_ratio:.3f}, accuracy={accuracy:.3f}")

    return {
        'structure_rate': structure_rate,
        'gate_ratio': gate_ratio,
        'basic_accuracy': accuracy if stage == 1 else 0.0,
        'accuracy': accuracy,
        'avg_think_len': avg_think_len,
        'avg_reward': avg_reward
    }

def should_transition(metrics, history):
    if metrics['structure_rate'] < 0.75 or metrics['gate_ratio'] < 3.0 or metrics['basic_accuracy'] < 0.4:
        return False

    if len(history['structure_rate']) < 3:
        return False

    recent_struct = history['structure_rate'][-3:]
    recent_gate = history['gate_ratio'][-3:]
    recent_acc = history['basic_accuracy'][-3:]

    stable_struct = max(recent_struct) - min(recent_struct) < 0.05
    stable_gate = max(recent_gate) - min(recent_gate) < 0.5
    stable_acc = max(recent_acc) - min(recent_acc) < 0.05

    return stable_struct and stable_gate and stable_acc