import os
import yaml
import torch
import wandb
from unsloth import FastLanguageModel, PatchFastRL, is_bfloat16_supported
from trl import GRPOConfig, GRPOTrainer
from datasets import load_dataset, Dataset
from model import SparseGatedModel, gumbel_sigmoid
from reward import custom_reward
from utils import preprocess_dataset

os.environ["WANDB_PROJECT"] = "latent-reasoning"
PatchFastRL("GRPO", FastLanguageModel)

class CustomGRPOTrainer(GRPOTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.progress_table = wandb.Table(columns=["step", "loss", "reward", "correctness", "token_penalty", "gate_penalty", "gate_mean", "gate_std"])

    def generate_and_compute_reward(self, batch):
        """Generate sequences and compute rewards with detailed logging."""
        input_ids = batch["input_ids"].to(self.model.device)
        attention_mask = batch["attention_mask"].to(self.model.device)
        ground_truths = batch["ground_truths"]
        prompt_len = input_ids.shape[1]

        outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.args.max_completion_length,
            temperature=self.args.temperature,
            num_return_sequences=self.args.num_generations,
            output_hidden_states=True,
            return_dict_in_generate=True,
        )

        sequences = outputs.sequences
        hidden_states = outputs.hidden_states
        batch_size = input_ids.shape[0]
        num_generations = self.args.num_generations
        seq_len = sequences.shape[1] - prompt_len

        gate_values_all_steps = []
        for step in range(seq_len):
            hs = hidden_states[step][-1][:, -1, :]
            gate_logits = self.model.gate_linear(hs)
            gate_val = gumbel_sigmoid(gate_logits, self.model.temperature, hard=False)
            gate_values_all_steps.append(gate_val.squeeze(-1))

        gate_values_per_sequence = [
            [gate_values_all_steps[t][i].item() for t in range(seq_len)]
            for i in range(batch_size * num_generations)
        ]

        completions = [seq.tolist() for seq in sequences]

        rewards = self.reward_funcs[0](
            completions=completions,
            gate_values_list=gate_values_per_sequence,
            tokenizer=self.processing_class,
            ground_truths=ground_truths,
            prompt_len=prompt_len,
            lambda_penalty=self.args.lambda_penalty,
            gate_penalty=self.args.gate_penalty,
            num_generations=num_generations
        )

        # Compute reward components for logging
        correctness = []
        token_penalties = []
        gate_penalties = []
        for k, (completion, gate_values) in enumerate(zip(completions, gate_values_per_sequence)):
            ground_truth = ground_truths[k // num_generations]
            completion_tensor = torch.tensor(completion)
            bot_id = self.processing_class.convert_tokens_to_ids("[bot]")
            eot_id = self.processing_class.convert_tokens_to_ids("[eot]")
            bot_idx = torch.argmax((completion_tensor == bot_id).float()).item() if (completion_tensor == bot_id).sum() > 0 else -1
            eot_idx = torch.argmax((completion_tensor == eot_id).float()).item() if (completion_tensor == eot_id).sum() > 0 else -1
            if not (bot_idx >= prompt_len and eot_idx > bot_idx):
                continue
            answer_tokens = completion[eot_idx + 1:] if eot_idx + 1 < len(completion) else []
            answer_text = self.processing_class.decode(answer_tokens, skip_special_tokens=True).strip()
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
            correctness.append(is_correct)
            token_penalties.append(self.args.lambda_penalty * num_tokens_in_reasoning)
            gate_penalties.append(self.args.gate_penalty * gate_penalty_avg)

        # Log metrics
        gate_values_flat = [val for seq in gate_values_per_sequence for val in seq]
        if gate_values_flat:
            gate_mean = torch.tensor(gate_values_flat).mean().item()
            gate_std = torch.tensor(gate_values_flat).std().item() if len(gate_values_flat) > 1 else 0.0
            gate_min = min(gate_values_flat)
            gate_max = max(gate_values_flat)
        else:
            gate_mean = gate_std = gate_min = gate_max = 0.0

        wandb.log({
            "batch_reward": sum(rewards) / len(rewards) if rewards else 0.0,
            "correctness": sum(correctness) / len(correctness) if correctness else 0.0,
            "token_penalty": sum(token_penalties) / len(token_penalties) if token_penalties else 0.0,
            "gate_penalty": sum(gate_penalties) / len(gate_penalties) if gate_penalties else 0.0,
            "gate_mean": gate_mean,
            "gate_std": gate_std,
            "gate_min": gate_min,
            "gate_max": gate_max,
            "gate_histogram": wandb.Histogram(gate_values_flat)
        })

        return completions, rewards

    def training_step(self, *args, **kwargs):
        """Log training progress to wandb table."""
        loss = super().training_step(*args, **kwargs)
        step = self.state.global_step
        batch_metrics = self.get_batch_metrics()
        self.progress_table.add_data(
            step,
            loss.item(),
            batch_metrics.get("batch_reward", 0.0),
            batch_metrics.get("correctness", 0.0),
            batch_metrics.get("token_penalty", 0.0),
            batch_metrics.get("gate_penalty", 0.0),
            batch_metrics.get("gate_mean", 0.0),
            batch_metrics.get("gate_std", 0.0)
        )
        wandb.log({"progress_table": self.progress_table})
        return loss

def main(config_path):
    """Train the sparse gated model using GRPO with enhanced monitoring."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    exp_name = config["output_dir"]
    if os.path.exists(exp_name) and len(os.listdir(exp_name)) > 0:
        print(f"Experiment {exp_name} already exists. Exiting...")
        return

    wandb.init(project="latent-reasoning", name=exp_name, config=config)
    wandb.save(config_path)

    model = SparseGatedModel(
        model_name="gpt2",
        hidden_size=768,
        embedding_dim=768,
        lora_rank=config["lora_rank"],
        temperature=config["gate_temperature"]
    )
    tokenizer = FastLanguageModel.from_pretrained(
        model_name="gpt2",
        max_seq_length=config["max_prompt_length"] + config["max_completion_length"]
    ).tokenizer
    tokenizer.add_special_tokens({'additional_special_tokens': ['[bot]', '[eot]']})
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    training_args = GRPOConfig(
        use_vllm=False,
        learning_rate=config["lr"],
        beta=config["beta"],
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=config["weight_decay"],
        warmup_ratio=config["warmup_ratio"],
        lr_scheduler_type=config["lr_scheduler_type"],
        optim=config["optimizer"],
        max_grad_norm=config["max_grad_norm"],
        logging_steps=1,
        bf16=is_bfloat16_supported(),
        fp16=not is_bfloat16_supported(),
        temperature=config["temperature"],
        num_generations=config["group_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        per_device_train_batch_size=config["per_device_train_batch_size"],
        max_prompt_length=config["max_prompt_length"],
        max_completion_length=config["max_completion_length"],
        num_train_epochs=config["num_train_epochs"],
        save_steps=config["save_steps"],
        save_total_limit=config["save_total_limit"],
        report_to="wandb",
        output_dir=exp_name,
        lambda_penalty=config["lambda_penalty"],
        gate_penalty=config["gate_penalty"]
    )

    dataset = preprocess_dataset(config["dataset"], config["data_dir"], "train")

    trainer = CustomGRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[custom_reward],
        args=training_args,
        train_dataset=dataset
    )

    trainer.train()
    model.save_pretrained(exp_name)
    tokenizer.save_pretrained(exp_name)
    wandb.save(os.path.join(exp_name, "*"))
    print(f"Model saved to {exp_name}")
    wandb.finish()

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python train.py <config_path>")
        sys.exit(1)
    main(sys.argv[1])