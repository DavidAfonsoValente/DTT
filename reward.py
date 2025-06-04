from typing import List
import torch

def custom_reward(completions: List[List[int]], gate_values_list: List[List[float]], tokenizer, ground_truths: List[str], prompt_len: int, lambda_penalty: float = 0.05, gate_penalty: float = 0.1, num_generations: int = 1) -> List[float]:
    """Reward function encouraging correct answers, minimal tokens, and proper gate values."""
    rewards = []
    bot_id = tokenizer.convert_tokens_to_ids("[bot]")
    eot_id = tokenizer.convert_tokens_to_ids("[eot]")

    for k, (completion, gate_values) in enumerate(zip(completions, gate_values_list)):
        ground_truth = ground_truths[k // num_generations]
        if not completion or not ground_truth:
            rewards.append(-1.0)
            continue

        completion_tensor = torch.tensor(completion)
        bot_mask = (completion_tensor == bot_id).float()
        eot_mask = (completion_tensor == eot_id).float()
        bot_idx = torch.argmax(bot_mask).item() if bot_mask.sum() > 0 else -1
        eot_idx = torch.argmax(eot_mask).item() if eot_mask.sum() > 0 else -1
        
        if not (bot_idx >= prompt_len and eot_idx > bot_idx):
            rewards.append(-1.0)
            continue

        answer_tokens = completion[eot_idx + 1:] if eot_idx + 1 < len(completion) else []
        answer_text = tokenizer.decode(answer_tokens, skip_special_tokens=True).strip()
        is_correct = float(answer_text == ground_truth.strip())
        num_tokens_in_reasoning = eot_idx - bot_idx - 1

        gate_penalty_sum = 0.0
        for i in range(prompt_len, len(completion)):
            gate_val = gate_values[i - prompt_len]
            if bot_idx < i <= eot_idx:
                gate_penalty_sum += (1.0 - gate_val) ** 2
            else:
                gate_penalty_sum += gate_val ** 2
        gate_penalty_avg = gate_penalty_sum / (len(completion) - prompt_len) if len(completion) > prompt_len else 0.0

        reward = is_correct - lambda_penalty * num_tokens_in_reasoning - gate_penalty * gate_penalty_avg
        rewards.append(reward)

    return rewards