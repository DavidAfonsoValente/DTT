import os
import torch
import yaml
import pandas as pd
import wandb
from transformers import GenerationConfig
from torch.utils.data import DataLoader

from model import SparseGatedModel
from utils import preprocess_dataset

@torch.no_grad()
def generate_for_evaluation(model, input_ids_b, attention_mask_b, max_gen_len, gen_config, eval_gumbel_hard):
    model.eval()
    batch_size = input_ids_b.shape[0]
    device = model.device
    
    generated_ids = input_ids_b.clone()
    current_attention_mask = attention_mask_b.clone()
    all_gates = [[] for _ in range(batch_size)]
    
    initial_outputs, h_t_prompt = model(
        input_ids=input_ids_b, attention_mask=current_attention_mask, use_cache=True,
        gumbel_hard_during_forward=eval_gumbel_hard
    )
    past_key_values = initial_outputs.past_key_values
    prev_h = h_t_prompt[:, -1:, :]
    next_token_logits = initial_outputs.logits[:, -1, :]

    for _ in range(max_gen_len):
        if gen_config.do_sample:
            probs = torch.softmax(next_token_logits / gen_config.temperature, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1)
        else:
            next_tokens = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
        
        generated_ids = torch.cat([generated_ids, next_tokens], dim=1)
        current_attention_mask = torch.cat([current_attention_mask, torch.ones_like(next_tokens, device=device)], dim=1)

        if (gen_config.eos_token_id is not None) and (next_tokens.squeeze(-1) == gen_config.eos_token_id).all():
            break

        outputs_step, h_t_step = model(
            input_ids=next_tokens, attention_mask=current_attention_mask, past_key_values=past_key_values,
            prev_step_last_hidden_state=prev_h, use_cache=True,
            gumbel_hard_during_forward=eval_gumbel_hard
        )
        
        step_gates = model.current_gate_values_for_batch.squeeze()
        for i in range(batch_size):
            all_gates[i].append(step_gates[i].item() if batch_size > 1 else step_gates.item())

        past_key_values = outputs_step.past_key_values
        prev_h = h_t_step
        next_token_logits = outputs_step.logits[:, -1, :]
        
    return generated_ids, all_gates

def _collate_fn_eval(batch, tokenizer):
    input_ids = [torch.tensor(item['input_ids']) for item in batch]
    attn_mask = [torch.tensor(item['attention_mask']) for item in batch]
    input_ids_padded = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    attn_mask_padded = torch.nn.utils.rnn.pad_sequence(attn_mask, batch_first=True, padding_value=0)
    gts = [item['ground_truths'] for item in batch]
    return {"input_ids": input_ids_padded, "attention_mask": attn_mask_padded, "ground_truths": gts}

def evaluate_model(eval_config_path):
    with open(eval_config_path, 'r') as f: config = yaml.safe_load(f)

    model_path = config["model_checkpoint_path"]
    eval_dir = config.get("eval_output_dir", os.path.join(model_path, "evaluation"))
    os.makedirs(eval_dir, exist_ok=True)

    exp_name = f"eval-{os.path.basename(model_path.rstrip('/'))}-{config['dataset_name']}"
    wandb.init(project=config.get("wandb_project_eval", "dtt-eval"), name=exp_name, config=config)

    model = SparseGatedModel(
        model_name=config["base_model_name_for_eval"],
        max_seq_length=config["max_prompt_length"] + config["max_completion_length"],
        hidden_size=config["hidden_size"], embedding_dim=config["embedding_dim"],
        lora_rank=config["lora_rank"], gate_temperature=config["eval_gate_temperature"],
        load_in_4bit=config.get("load_in_4bit_eval", True),
        lora_target_modules=config.get("lora_target_modules_eval")
    )
    model.model.load_adapter(model_path)
    tokenizer = model.tokenizer

    special_tokens = {'additional_special_tokens': ['[bot]', '[eot]']}
    if tokenizer.add_special_tokens(special_tokens) > 0:
        model.model.resize_token_embeddings(len(tokenizer))
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); model.to(device)

    eval_dataset = preprocess_dataset(
        config["dataset_name"], config["data_dir"], tokenizer,
        config["max_prompt_length"], config.get("dataset_split_eval", "test")
    )
    eval_loader = DataLoader(eval_dataset, batch_size=config["eval_batch_size"], collate_fn=lambda b: _collate_fn_eval(b, tokenizer))

    gen_config = GenerationConfig(
        max_new_tokens=config["max_completion_length"], temperature=config["sampling_temperature_eval"],
        do_sample=config["sampling_temperature_eval"] > 0, pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    results, total_correct = [], 0
    bot_id, eot_id = tokenizer.convert_tokens_to_ids(["[bot]", "[eot]"])

    for batch in eval_loader:
        ids_b, mask_b, gts_b = batch["input_ids"].to(device), batch["attention_mask"].to(device), batch["ground_truths"]
        p_len = ids_b.shape[1]
        gen_ids, gen_gates = generate_for_evaluation(model, ids_b, mask_b, config["max_completion_length"], gen_config, config.get("eval_gumbel_hard", True))

        for i in range(gen_ids.shape[0]):
            full_ids, gates = gen_ids[i].tolist(), gen_gates[i]
            t = torch.tensor(full_ids); b_idxs, e_idxs = (t[p_len:]==bot_id).nonzero()+p_len, (t[p_len:]==eot_id).nonzero()+p_len
            b_idx, e_idx = -1, -1
            for b in b_idxs:
                for e in e_idxs:
                    if e > b: b_idx, e_idx = b.item(), e.item(); break
                if b_idx != -1: break
            
            r_txt, pred_txt, correct = "", "", 0.0
            if b_idx != -1:
                r_txt = tokenizer.decode(full_ids[b_idx+1:e_idx], skip_special_tokens=True).strip()
                pred_txt = tokenizer.decode(full_ids[e_idx+1:], skip_special_tokens=True).strip()
                if pred_txt == gts_b[i].strip(): correct = 1.0
            total_correct += correct

            r_g, nr_g = [], []
            for j, g in enumerate(gates):
                if b_idx != -1 and b_idx < p_len + j <= e_idx: r_g.append(g)
                else: nr_g.append(g)
            
            results.append({
                "prompt": tokenizer.decode(full_ids[:p_len], skip_special_tokens=True),
                "reasoning_text": r_txt, "predicted_answer": pred_txt, "ground_truth": gts_b[i],
                "is_correct": correct, "gate_mean_reasoning": torch.tensor(r_g).mean().item() if r_g else None,
                "gate_mean_non_reasoning": torch.tensor(nr_g).mean().item() if nr_g else None,
            })

    accuracy = total_correct / len(results) if results else 0.0
    print(f"Final Accuracy: {accuracy:.4f} ({int(total_correct)}/{len(results)})")
    
    if wandb.run:
        wandb.log({"evaluation/accuracy": accuracy})
        df = pd.DataFrame(results)
        wandb.log({"evaluation/results_table": wandb.Table(dataframe=df)})
        wandb.finish()
        
    df.to_csv(os.path.join(eval_dir, "evaluation_results.csv"), index=False)
    print(f"Evaluation results saved to {eval_dir}/evaluation_results.csv")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2: print("Usage: python eval.py <path_to_eval_config.yaml>"); sys.exit(1)
    evaluate_model(sys.argv[1])
