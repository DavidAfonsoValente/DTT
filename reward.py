import torch

def custom_reward(completions_ids, gate_values_list, tokenizer, ground_truths, prompt_len, lambda_penalty, gate_penalty_coeff, num_generations, **kwargs):
    """
    Computes reward based on correctness, token efficiency, and gate sparsity.
    
    Args:
        completions_ids: List of generated token ID sequences.
        gate_values_list: List of gate values for each generated sequence.
        tokenizer: Tokenizer for decoding sequences.
        ground_truths: List of correct answers.
        prompt_len: Length of the prompt in tokens.
        lambda_penalty: Coefficient for token efficiency penalty.
        gate_penalty_coeff: Coefficient for gate sparsity penalty.
        num_generations: Number of generations per prompt.
    
    Returns:
        List of reward values for each completion.
    """
    rewards = []
    bot_id = tokenizer.convert_tokens_to_ids("[bot]")
    eot_id = tokenizer.convert_tokens_to_ids("[eot]")
    
    if bot_id is None or eot_id is None:
        raise ValueError("[bot] or [eot] tokens not found in tokenizer.")
    
    for k, (full_completion_token_ids, gate_values_for_generated_tokens) in enumerate(zip(completions_ids, gate_values_list)):
        original_batch_idx = k // num_generations
        ground_truth_answer_str = ground_truths[original_batch_idx]
        
        completion_tensor = torch.tensor(full_completion_token_ids, dtype=torch.long)
        bot_indices = (completion_tensor[prompt_len:] == bot_id).nonzero(as_tuple=True)[0] + prompt_len
        eot_indices = (completion_tensor[prompt_len:] == eot_id).nonzero(as_tuple=True)[0] + prompt_len
        
        current_bot_idx, current_eot_idx = -1, -1
        for b_idx in bot_indices:
            if b_idx >= prompt_len:
                for e_idx in eot_indices:
                    if e_idx > b_idx:
                        current_bot_idx, current_eot_idx = b_idx.item(), e_idx.item()
                        break
                if current_bot_idx != -1:
                    break
        
        is_correct = 0.0
        if current_bot_idx != -1 and current_eot_idx != -1:
            answer_token_ids = full_completion_token_ids[current_eot_idx + 1:]
            predicted_answer_str = tokenizer.decode(answer_token_ids, skip_special_tokens=True).strip()
            if predicted_answer_str == ground_truth_answer_str.strip():
                is_correct = 1.0
        else:
            is_correct = -0.5
        
        num_tokens_in_reasoning = 0
        if current_bot_idx != -1 and current_eot_idx != -1:
            num_tokens_in_reasoning = current_eot_idx - (current_bot_idx + 1)
            if num_tokens_in_reasoning < 0:
                num_tokens_in_reasoning = 0
        
        token_eff_penalty = lambda_penalty * num_tokens_in_reasoning
        
        gate_penalty_sum = 0.0
        if gate_values_for_generated_tokens:
            for gen_token_idx, gate_val in enumerate(gate_values_for_generated_tokens):
                actual_token_idx_in_full_seq = prompt_len + gen_token_idx
                if current_bot_idx != -1 and current_eot_idx != -1 and current_bot_idx < actual_token_idx_in_full_seq <= current_eot_idx:
                    gate_penalty_sum += (1.0 - gate_val) ** 2
                else:
                    gate_penalty_sum += gate_val ** 2
            avg_gate_penalty = gate_penalty_sum / len(gate_values_for_generated_tokens)
        else:
            avg_gate_penalty = 1.0 if len(full_completion_token_ids) > prompt_len else 0.0
        
        total_reward = is_correct - token_eff_penalty - (gate_penalty_coeff * avg_gate_penalty)
        rewards.append(total_reward)
    
    return rewards