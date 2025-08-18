import torch

def compute_reward(completion_ids, gates, tokenizer, answer_gt, bot_id, eot_id):
    bot_pos = (completion_ids == bot_id).nonzero(as_tuple=True)[0]
    eot_pos = (completion_ids == eot_id).nonzero(as_tuple=True)[0]
    has_span = len(bot_pos) > 0 and len(eot_pos) > 0 and bot_pos[0] < eot_pos[0]
    
    r_struct = 0.2 if has_span else -1.0
    
    if has_span:
        think_len = eot_pos[0] - bot_pos[0] - 1
        r_eff = -0.01 * think_len.item()
        pred_answer = tokenizer.decode(completion_ids[eot_pos[0] + 1:]).strip()
    else:
        r_eff = 0.0
        pred_answer = tokenizer.decode(completion_ids).strip()
    
    if 'gsm8k' in str(tokenizer).lower():
        try:
            pred_num = float(pred_answer.split('####')[-1].strip() if '####' in pred_answer else pred_answer)  # GSM format
            gt_num = float(answer_gt.split('####')[-1].strip())
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

    components = [r_struct, r_corr, r_eff, r_gate.item()]
    clipped = [min(max(c, -1.5), 1.5) for c in components]
    return sum(clipped)