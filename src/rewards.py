# src/rewards.py
import torch

def compute_reward(completion_ids, gates, tokenizer, answer_gt, bot_id, eot_id, dataset, dummy_id, weights=None):
    if weights is None:
        weights = {'struct': 1.0, 'corr': 1.0, 'eff': 1.0, 'gate': 1.0}
    bot_pos = (completion_ids == bot_id).nonzero(as_tuple=True)[0]
    eot_pos = (completion_ids == eot_id).nonzero(as_tuple=True)[0]
    has_span = len(bot_pos) > 0 and len(eot_pos) > 0 and bot_pos[0] < eot_pos[0]
    
    r_struct = 0.2 if has_span else -1.0
    
    if has_span:
        span_ids = completion_ids[bot_pos[0] + 1:eot_pos[0]]
        think_len = (span_ids != dummy_id).sum().item()  # Surface tokens only
        r_eff = -0.01 * think_len
        pred_ids = [id.item() for id in completion_ids[eot_pos[0] + 1:] if id != dummy_id]
        pred_answer = tokenizer.decode(pred_ids).strip()
    else:
        r_eff = 0.0
        pred_ids = [id.item() for id in completion_ids if id != dummy_id]
        pred_answer = tokenizer.decode(pred_ids).strip()
    
    if dataset == 'gsm8k':
        try:
            pred_num = float(pred_answer.split('####')[-1].strip() if '####' in pred_answer else pred_answer)
            gt_num = float(answer_gt)
            r_corr = 1.0 if abs(pred_num - gt_num) < 1e-3 else -0.5
        except:
            r_corr = -0.5
    else:
        r_corr = 1.0 if pred_answer == answer_gt.strip() else -0.5
    
    if has_span:
        inner_gates = gates[bot_pos[0] + 1:eot_pos[0]]
        outer_gates = torch.cat([gates[:bot_pos[0]], gates[eot_pos[0]:]])
        g_in = inner_gates.mean() if len(inner_gates) > 0 else torch.tensor(0.0)
        g_out = outer_gates.mean() if len(outer_gates) > 0 else torch.tensor(0.0)
        r_gate = 0.5 * g_in - 0.2 * g_out
    else:
        r_gate = -0.2
    
    components = [r_struct, r_corr, r_eff, r_gate.item() if isinstance(r_gate, torch.Tensor) else r_gate]
    clipped = [min(max(c, -1.5), 1.5) for c in components]
    total_reward = sum(weights[k] * clipped[i] for i, k in enumerate(['struct', 'corr', 'eff', 'gate']))
    
    # Return dict for logging components
    return {
        'total': total_reward,
        'struct': clipped[0],
        'corr': clipped[1],
        'eff': clipped[2],
        'gate': clipped[3]
    }