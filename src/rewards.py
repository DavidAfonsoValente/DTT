# src/rewards.py
import torch
import re

def find_first_valid_span(completion_ids, bot_id, eot_id):
    bot_positions = (completion_ids == bot_id).nonzero(as_tuple=True)[0]
    eot_positions = (completion_ids == eot_id).nonzero(as_tuple=True)[0]
    for bot_pos in bot_positions:
        for eot_pos in eot_positions:
            if bot_pos < eot_pos:
                return bot_pos.item(), eot_pos.item()
    return -1, -1

def compute_stage1_reward(completion_ids, gates, tokenizer, answer_gt, bot_id, eot_id, dataset):
    bot_pos, eot_pos = find_first_valid_span(completion_ids, bot_id, eot_id)

    if bot_pos == -1 and eot_pos == -1:
        r_struct = -2.0
    elif bot_pos == -1 or eot_pos == -1:
        r_struct = 0.5
    elif eot_pos <= bot_pos:
        r_struct = -1.0
    else:
        think_len = eot_pos - bot_pos - 1
        if think_len <= 0:
            r_struct = -1.0  # Downgrade for empty or invalid span
        else:
            r_struct = 2.0

    r_struct = max(-1.5, min(1.5, r_struct))  # Clip per component

    has_span = bot_pos != -1 and eot_pos != -1 and bot_pos < eot_pos and think_len > 0  # Updated condition

    if has_span:
        pred_answer = tokenizer.decode(completion_ids[eot_pos + 1 :]).strip()
    else:
        pred_answer = tokenizer.decode(completion_ids).strip()

    if dataset == 'gsm8k':
        try:
            pred_num = float(re.findall(r'[\d\.-]+$', pred_answer)[-1]) if re.findall(r'[\d\.-]+$', pred_answer) else 0.0
            gt_num = float(answer_gt)
            is_approx_correct = abs(pred_num - gt_num) < 10.0
        except:
            is_approx_correct = False
    elif dataset == 'prontoqa':
        is_approx_correct = pred_answer.lower().strip() in ['true', 'false']
    elif dataset == 'prosqa':
        is_approx_correct = len(pred_answer.strip()) > 0
    else:
        is_approx_correct = False

    r_basic = 1.0 if is_approx_correct else 0.0
    r_basic = max(-1.5, min(1.5, r_basic))

    if has_span:
        inner_gates = gates[bot_pos + 1 : eot_pos]
        outer_gates_before = gates[:bot_pos] if bot_pos > 0 else torch.tensor([], device=gates.device)
        outer_gates_after = gates[eot_pos:] if eot_pos < len(gates) else torch.tensor([], device=gates.device)
        outer_gates = torch.cat([outer_gates_before, outer_gates_after])
        g_in = inner_gates.mean().item() if len(inner_gates) > 0 else 0.0
        g_out = outer_gates.mean().item() if len(outer_gates) > 0 else 0.0
        r_gate = 1.5 * g_in - 0.5 * g_out
    else:
        r_gate = -0.5

    r_gate = max(-1.5, min(1.5, r_gate))

    total = r_struct + r_gate + 0.3 * r_basic
    return {
        'total': total,
        'struct': r_struct,
        'basic': r_basic,
        'gate': r_gate
    }

def compute_stage2_reward(completion_ids, gates, tokenizer, answer_gt, bot_id, eot_id, dataset):
    bot_pos, eot_pos = find_first_valid_span(completion_ids, bot_id, eot_id)

    if bot_pos == -1 and eot_pos == -1:
        r_struct = -2.0
    elif bot_pos == -1 or eot_pos == -1:
        r_struct = 0.5
    elif eot_pos <= bot_pos:
        r_struct = -1.0
    else:
        think_len = eot_pos - bot_pos - 1
        if think_len <= 0:
            r_struct = -1.0  # Downgrade for empty or invalid span
        else:
            r_struct = 2.0
    r_struct *= 0.6
    r_struct = max(-1.5, min(1.5, r_struct))

    has_span = bot_pos != -1 and eot_pos != -1 and bot_pos < eot_pos and think_len > 0  # Updated condition

    if has_span:
        think_len = eot_pos - bot_pos - 1
        r_eff = -0.03 * think_len - 0.01 * max(0, think_len - 10)
        r_eff = max(-1.5, min(1.5, r_eff))
        pred_answer = tokenizer.decode(completion_ids[eot_pos + 1 :]).strip()
    else:
        r_eff = 0.0
        pred_answer = tokenizer.decode(completion_ids).strip()

    if dataset == 'gsm8k':
        try:
            pred_num = float(re.findall(r'[\d\.-]+$', pred_answer)[-1]) if re.findall(r'[\d\.-]+$', pred_answer) else 0.0
            gt_num = float(answer_gt)
            diff = abs(pred_num - gt_num)
            if diff < 1e-3:
                r_corr = 3.0
            elif diff < 10.0:
                r_corr = 1.0
            else:
                r_corr = -1.5
        except:
            r_corr = -1.5
    elif dataset == 'prontoqa':
        if pred_answer.lower().strip() == answer_gt.lower().strip():
            r_corr = 3.0
        elif pred_answer.lower().strip() in ['true', 'false']:
            r_corr = 1.0
        else:
            r_corr = -1.5
    elif dataset == 'prosqa':
        if pred_answer.strip() == answer_gt.strip():
            r_corr = 3.0
        elif len(pred_answer.strip()) > 0:
            r_corr = 1.0
        else:
            r_corr = -1.5
    else:
        r_corr = -1.5

    r_corr = max(-1.5, min(1.5, r_corr))

    if has_span:
        inner_gates = gates[bot_pos + 1 : eot_pos]
        outer_gates_before = gates[:bot_pos] if bot_pos > 0 else torch.tensor([], device=gates.device)
        outer_gates_after = gates[eot_pos:] if eot_pos < len(gates) else torch.tensor([], device=gates.device)
        outer_gates = torch.cat([outer_gates_before, outer_gates_after])
        g_in = inner_gates.mean().item() if len(inner_gates) > 0 else 0.0
        g_out = outer_gates.mean().item() if len(outer_gates) > 0 else 0.0
        r_gate = (1.5 * g_in - 0.5 * g_out) * 0.8
    else:
        r_gate = -0.5

    r_gate = max(-1.5, min(1.5, r_gate))

    total = r_struct + r_corr + r_eff + r_gate
    return {
        'total': total,
        'struct': r_struct,
        'corr': r_corr,
        'eff': r_eff,
        'gate': r_gate
    }