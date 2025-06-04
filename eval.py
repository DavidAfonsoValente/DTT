import os
import yaml
import torch
import wandb
from datasets import load_dataset
from model import SparseGatedModel, gumbel_sigmoid
from utils import preprocess_dataset, process_answer

def evaluate(config_path):
    """Evaluate the trained model with enhanced metrics."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    checkpoint_path = config["checkpoint_path"]
    wandb.init(project="latent-reasoning", name=f"eval-{os.path.basename(checkpoint_path)}", config=config)
    wandb.save(config_path)

    model = SparseGatedModel(
        model_name="gpt2",
        hidden_size=768,
        embedding_dim=768,
        lora_rank=config["lora_rank"],
        temperature=config["gate_temperature"]
    )
    model.load_state_dict(torch.load(os.path.join(checkpoint_path, "pytorch_model.bin")))
    tokenizer = FastLanguageModel.from_pretrained("gpt2").tokenizer
    tokenizer.add_special_tokens({'additional_special_tokens': ['[bot]', '[eot]']})
    model.eval()

    dataset = preprocess_dataset(config["dataset"], config["data_dir"], "test")
    
    correct = 0
    total_latent_steps = 0
    total = 0
    avg_gate_inside = 0.0
    avg_gate_outside = 0.0
    total_inside = 0
    total_outside = 0
    hidden_states_all = []
    gate_values_all = []
    reasoning_lengths = []
    gate_values_flat = []

    # Per-example metrics table
    eval_table = wandb.Table(columns=["question", "generated", "answer", "ground_truth", "is_correct", "latent_steps", "gate_inside", "gate_outside"])

    for i in range(0, len(dataset), config["batch_size"]):
        batch = dataset[i:i + config["batch_size"]]
        input_ids = tokenizer(batch['question'], return_tensors="pt", padding=True).input_ids.to(model.device)
        attention_mask = tokenizer(batch['question'], return_tensors="pt", padding=True).attention_mask.to(model.device)
        prompt_len = input_ids.shape[1]

        try:
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=config["max_length"],
                output_hidden_states=True,
                return_dict_in_generate=True
            )
        except Exception as e:
            print(f"Generation failed for batch {i}: {e}")
            continue

        sequences = outputs.sequences
        hidden_states = outputs.hidden_states
        seq_len = sequences.shape[1] - prompt_len

        gate_values_batch = []
        for step in range(seq_len):
            hs = hidden_states[step][-1][:, -1, :]
            gate_logits = model.gate_linear(hs)
            gate_val = gumbel_sigmoid(gate_logits, model.temperature, hard=False)
            gate_values_batch.append(gate_val.squeeze(-1))

        gate_values_list = [torch.stack([gate_values_batch[t][i] for t in range(seq_len)]).tolist() for i in range(sequences.shape[0])]

        hidden_states_all.extend([hs.cpu() for hs in hidden_states])
        gate_values_all.extend(gate_values_list)

        for j, gen_ids in enumerate(sequences):
            gen_text = tokenizer.decode(gen_ids, skip_special_tokens=False)
            bot_token_id = tokenizer.convert_tokens_to_ids("[bot]")
            eot_token_id = tokenizer.convert_tokens_to_ids("[eot]")
            try:
                bot_idx = gen_ids.tolist().index(bot_token_id)
                eot_idx = gen_ids.tolist().index(eot_token_id, bot_idx + 1)
                latent_steps = eot_idx - bot_idx - 1
                answer_text = tokenizer.decode(gen_ids[eot_idx + 1:], skip_special_tokens=True).strip()
                is_correct = process_answer(answer_text, batch['answer'][j], config["dataset"])
                if is_correct:
                    correct += 1
                total_latent_steps += latent_steps
                reasoning_lengths.append(latent_steps)
                total += 1

                gate_values = gate_values_list[j]
                gate_inside = 0.0
                gate_outside = 0.0
                count_inside = 0
                count_outside = 0
                for k in range(prompt_len, len(gen_ids)):
                    gate_val = gate_values[k - prompt_len]
                    gate_values_flat.append(gate_val)
                    if bot_idx < k <= eot_idx:
                        gate_inside += gate_val
                        count_inside += 1
                        avg_gate_inside += gate_val
                        total_inside += 1
                    else:
                        gate_outside += gate_val
                        count_outside += 1
                        avg_gate_outside += gate_val
                        total_outside += 1

                gate_inside_avg = gate_inside / count_inside if count_inside > 0 else 0.0
                gate_outside_avg = gate_outside / count_outside if count_outside > 0 else 0.0

                eval_table.add_data(
                    batch['question'][j],
                    gen_text,
                    answer_text,
                    batch['answer'][j],
                    is_correct,
                    latent_steps,
                    gate_inside_avg,
                    gate_outside_avg
                )
            except ValueError as e:
                print(f"Invalid sequence in batch {i}, example {j}: {e}")
                continue

        if total > 0:
            wandb.log({
                "eval_accuracy": correct / total,
                "avg_latent_steps": total_latent_steps / total,
                "avg_gate_inside": avg_gate_inside / total_inside if total_inside > 0 else 0.0,
                "avg_gate_outside": avg_gate_outside / total_outside if total_outside > 0 else 0.0,
                "gate_histogram": wandb.Histogram(gate_values_flat),
                "reasoning_length_histogram": wandb.Histogram(reasoning_lengths),
                "eval_table": eval_table
            })

    torch.save(hidden_states_all, os.path.join(checkpoint_path, "hidden_states.pt"))
    torch.save(gate_values_all, os.path.join(checkpoint_path, "gate_values.pt"))
    print(f"Hidden states saved to {checkpoint_path}/hidden_states.pt")
    print(f"Gate values saved to {checkpoint_path}/gate_values.pt")
    
    wandb.finish()

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python eval.py <config_path>")
        sys.exit(1)
    evaluate(sys.argv[1])