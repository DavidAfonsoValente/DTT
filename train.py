import os
import sys
import yaml
import torch
import wandb
from dataclasses import dataclass, field
from transformers import GenerationConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer

from model import SparseGatedModel
from reward import custom_reward
from utils import preprocess_dataset


@dataclass
class CustomGRPOConfig(GRPOConfig):
    """
    Custom GRPO configuration class to hold additional parameters for the
    gating mechanism and custom reward function.
    """
    initial_gate_temperature: float = field(default=1.0, metadata={"help": "Initial temperature for the Gumbel-Sigmoid gate."})
    min_gate_temperature: float = field(default=0.1, metadata={"help": "Minimum temperature for the Gumbel-Sigmoid gate."})
    gumbel_hard_generation: bool = field(default=False, metadata={"help": "Whether to use the hard version of Gumbel-Sigmoid during generation in training."})
    lambda_penalty: float = field(default=0.01, metadata={"help": "Coefficient for the token efficiency penalty in the reward."})
    gate_penalty_coeff: float = field(default=0.05, metadata={"help": "Coefficient for the gate sparsity penalty in the reward."})

class CustomGRPOTrainer(GRPOTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        if self.accelerator.is_main_process:
            self.progress_table = wandb.Table(columns=[
                "step", "loss", "reward_mean", "correctness_mean", "token_penalty_mean",
                "gate_penalty_mean", "gate_mean_overall", "gate_mean_reasoning", "gate_mean_non_reasoning",
                "gumbel_temp_model"
            ])
        
        train_batch_size = self.args.per_device_train_batch_size * \
                           self.accelerator.num_processes * \
                           self.args.gradient_accumulation_steps
        
        if len(self.train_dataset) > 0 and train_batch_size > 0:
            unique_prompts_per_optim_step = self.args.per_device_train_batch_size * \
                                            self.accelerator.num_processes * \
                                            self.args.gradient_accumulation_steps
            self.total_steps_for_annealing = (self.args.num_train_epochs * len(self.train_dataset)) // unique_prompts_per_optim_step
        else:
            self.total_steps_for_annealing = self.args.max_steps if self.args.max_steps > 0 else 1000

    def _anneal_gumbel_temperature(self):
        if self.total_steps_for_annealing <= 0: return self.args.initial_gate_temperature
        progress = min(1.0, self.state.global_step / self.total_steps_for_annealing)
        annealed_temp = self.args.initial_gate_temperature - \
                        (self.args.initial_gate_temperature - self.args.min_gate_temperature) * progress
        self.model.gate_temperature = max(self.args.min_gate_temperature, annealed_temp)
        return self.model.gate_temperature

    @torch.no_grad()
    def _custom_generate_sequences(self, input_ids, attention_mask):
        self.model.eval()
        batch_size, prompt_len = input_ids.shape
        device = self.model.device
        
        gen_config = GenerationConfig(
            max_new_tokens=self.args.max_completion_length,
            temperature=self.args.temperature,
            do_sample=self.args.temperature > 0,
            pad_token_id=self.processing_class.pad_token_id,
            eos_token_id=self.processing_class.eos_token_id,
        )
        
        current_gumbel_model_temp = self._anneal_gumbel_temperature()

        generated_ids = input_ids.clone()
        current_full_attention_mask = attention_mask.clone()
        all_gate_values = [[] for _ in range(batch_size)]
        
        initial_lm_outputs, h_t_after_prompt = self.model(
            input_ids=input_ids, attention_mask=current_full_attention_mask, use_cache=True,
            gumbel_hard_during_forward=self.args.gumbel_hard_generation,
            return_last_hidden_state=True # Pass flag to get hidden state
        )
        past_key_values = initial_lm_outputs.past_key_values
        prev_h_for_next_step = h_t_after_prompt[:, -1:, :]
        next_token_logits = initial_lm_outputs.logits[:, -1, :]

        for _ in range(self.args.max_completion_length):
            if gen_config.do_sample:
                probs = torch.softmax(next_token_logits / gen_config.temperature, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1)
            else:
                next_tokens = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
            
            generated_ids = torch.cat([generated_ids, next_tokens], dim=1)
            current_full_attention_mask = torch.cat(
                [current_full_attention_mask, torch.ones_like(next_tokens, device=device)], dim=1
            )

            if (gen_config.eos_token_id is not None) and (next_tokens.squeeze(-1) == gen_config.eos_token_id).all():
                break

            lm_outputs_step, h_t_current_step = self.model(
                input_ids=next_tokens, attention_mask=current_full_attention_mask, past_key_values=past_key_values,
                prev_step_last_hidden_state=prev_h_for_next_step, use_cache=True,
                gumbel_hard_during_forward=self.args.gumbel_hard_generation,
                return_last_hidden_state=True # Pass flag to get hidden state
            )
            
            step_gate_values = self.model.current_gate_values_for_batch.squeeze()
            for i in range(batch_size):
                all_gate_values[i].append(step_gate_values[i].item() if batch_size > 1 else step_gate_values.item())

            past_key_values = lm_outputs_step.past_key_values
            prev_h_for_next_step = h_t_current_step
            next_token_logits = lm_outputs_step.logits[:, -1, :]
        
        self.model.train()
        return generated_ids, all_gate_values, current_gumbel_model_temp

    # The rest of the trainer methods (generate_and_compute_reward, etc.) do not need changes
    def generate_and_compute_reward(self, batch):
        # This method remains the same
        input_ids, attention_mask = batch["input_ids"], batch["attention_mask"]
        ground_truths = batch["ground_truths"]
        prompt_len = input_ids.shape[1]
        
        expanded_input_ids = input_ids.repeat_interleave(self.args.num_generations, dim=0)
        expanded_attention_mask = attention_mask.repeat_interleave(self.args.num_generations, dim=0)
        
        sequences_ids, gate_values_list, current_gumbel_temp = self._custom_generate_sequences(
            expanded_input_ids.to(self.model.device), expanded_attention_mask.to(self.model.device)
        )
        
        completions_decoded = [self.processing_class.decode(s, skip_special_tokens=True) for s in sequences_ids]
        completions_ids = [s.tolist() for s in sequences_ids]

        rewards = self.reward_funcs[0](
            completions_ids=completions_ids, gate_values_list=gate_values_list,
            tokenizer=self.processing_class, ground_truths=ground_truths,
            prompt_len=prompt_len, lambda_penalty=self.args.lambda_penalty,
            gate_penalty_coeff=self.args.gate_penalty_coeff, num_generations=self.args.num_generations
        )
        
        if self.accelerator.is_main_process and self.state.global_step % self.args.logging_steps == 0:
            metrics = self._calculate_and_log_metrics(
                completions_ids, gate_values_list, rewards, ground_truths, prompt_len, current_gumbel_temp
            )
            self._log_to_progress_table(self.state.global_step, metrics, self.state.loss if hasattr(self.state, 'loss') else None)

        return completions_decoded, rewards

    def _calculate_and_log_metrics(self, completions, gates, rewards, gts, p_len, gumbel_temp):
        # This method remains the same
        metrics, scores, t_pens, g_pens, all_g, r_g, nr_g = {}, [], [], [], [], [], []
        bot_id, eot_id = self.processing_class.convert_tokens_to_ids(["[bot]", "[eot]"])

        for k, (ids, gen_gates) in enumerate(zip(completions, gates)):
            gt_ans = gts[k // self.args.num_generations]
            t = torch.tensor(ids); b_idxs, e_idxs = (t[p_len:]==bot_id).nonzero()+p_len, (t[p_len:]==eot_id).nonzero()+p_len
            b_idx, e_idx = -1, -1
            for b in b_idxs:
                for e in e_idxs:
                    if e > b: b_idx, e_idx = b.item(), e.item(); break
                if b_idx != -1: break
            
            correct = 0.0
            if b_idx != -1:
                pred = self.processing_class.decode(ids[e_idx+1:], skip_special_tokens=True).strip()
                if pred == gt_ans.strip(): correct = 1.0
            scores.append(correct)

            r_toks = (e_idx - b_idx - 1) if b_idx != -1 and e_idx > b_idx else 0
            t_pens.append(self.args.lambda_penalty * r_toks)
            
            g_pen_sum = 0
            if gen_gates:
                for i, g in enumerate(gen_gates):
                    all_g.append(g); idx = p_len + i
                    if b_idx != -1 and b_idx < idx <= e_idx:
                        g_pen_sum += (1.0 - g)**2; r_g.append(g)
                    else:
                        g_pen_sum += g**2; nr_g.append(g)
                g_pens.append(self.args.gate_penalty_coeff * (g_pen_sum / len(gen_gates)))
        
        metrics.update({
            "reward_mean": torch.tensor(rewards).mean().item(), "correctness_mean": torch.tensor(scores).mean().item(),
            "token_penalty_mean": torch.tensor(t_pens).mean().item(), "gate_penalty_mean": torch.tensor(g_pens).mean().item(),
            "gumbel_temp_model": gumbel_temp, "gate_mean_overall": torch.tensor(all_g).mean().item() if all_g else 0,
            "gate_mean_reasoning": torch.tensor(r_g).mean().item() if r_g else 0,
            "gate_mean_non_reasoning": torch.tensor(nr_g).mean().item() if nr_g else 0,
        })
        if all_g: metrics["gate_histogram_overall"] = wandb.Histogram(all_g)
        wandb.log({f"train/{k}": v for k, v in metrics.items() if "hist" not in k})
        if "gate_histogram_overall" in metrics: wandb.log({"train/gate_histogram_overall": metrics["gate_histogram_overall"]})
        return metrics

    def _log_to_progress_table(self, step, metrics, loss):
        # This method remains the same
        if self.accelerator.is_main_process:
            self.progress_table.add_data(
                step, loss.item() if loss else float('nan'), metrics.get("reward_mean", 0), metrics.get("correctness_mean", 0),
                metrics.get("token_penalty_mean", 0), metrics.get("gate_penalty_mean", 0), metrics.get("gate_mean_overall", 0),
                metrics.get("gate_mean_reasoning", 0), metrics.get("gate_mean_non_reasoning", 0), metrics.get("gumbel_temp_model", 0)
            )
            if step > 0 and step % (self.args.logging_steps * 10) == 0:
                wandb.log({"train_progress_summary": self.progress_table})

def main(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    exp_name = config["output_dir"]
    if os.path.exists(exp_name) and os.listdir(exp_name) and not config.get("overwrite_output_dir"):
        print(f"Directory {exp_name} exists and is not empty. Exiting.")
        return

    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        wandb.init(project=config.get("wandb_project", "dtt-project"), name=exp_name, config=config)
        wandb.save(config_path)

    # --- Corrected Model Initialization ---
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        config["model_name"],
        quantization_config=bnb_config,
        #device_map="auto",
    )

    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])

    lora_config = LoraConfig(
        r=config["lora_rank"],
        lora_alpha=config["lora_rank"] * 2,
        target_modules=config.get("lora_target_modules"),
        lora_dropout=0.05,
        bias="none",
    )

    model = SparseGatedModel(
        model=base_model,
        peft_config=lora_config,
        hidden_size=config["hidden_size"],
        embedding_dim=config["embedding_dim"],
        gate_temperature=config["initial_gate_temperature"],
    )
    # --- End of Corrected Model Initialization ---

    special_tokens = {'additional_special_tokens': ['[bot]', '[eot]']}
    if tokenizer.add_special_tokens(special_tokens) > 0:
        model.resize_token_embeddings(len(tokenizer))
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    training_args = CustomGRPOConfig(
        output_dir=exp_name,
        per_device_train_batch_size=config["per_device_train_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        num_train_epochs=config["num_train_epochs"],
        max_steps=config.get("max_steps", -1),
        save_steps=config["save_steps"],
        save_total_limit=config["save_total_limit"],
        learning_rate=config["lr"],
        lr_scheduler_type=config["lr_scheduler_type"],
        warmup_ratio=config["warmup_ratio"],
        optim=config.get("optimizer", "adamw_torch"),
        max_grad_norm=config["max_grad_norm"],
        logging_steps=config.get("logging_steps", 10),
        seed=config.get("seed", 42),
        report_to="wandb" if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0 else None,
        remove_unused_columns=False,
        bf16=config.get("use_bf16", False),
        fp16=config.get("use_fp16", False),
        
        # Standard GRPO arguments
        beta=config["beta_grpo"],
        temperature=config["sampling_temperature_grpo"],
        num_generations=config["group_size_grpo"],
        max_prompt_length=config["max_prompt_length"],
        max_completion_length=config["max_completion_length"],
        
        # Custom arguments
        initial_gate_temperature=config["initial_gate_temperature"],
        min_gate_temperature=config.get("min_gate_temperature", 0.1),
        gumbel_hard_generation=config.get("gumbel_hard_generation_train", False),
        lambda_penalty=config["lambda_penalty"],
        gate_penalty_coeff=config["gate_penalty_coeff"],
    )

    train_dataset = preprocess_dataset(
        config["dataset_name"], config["data_dir"], tokenizer, config["max_prompt_length"], "train"
    )

    trainer = CustomGRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        reward_funcs=[custom_reward]
    )
    trainer.train(resume_from_checkpoint=config.get("resume_from_checkpoint", None))

    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        # Saving the final model
        model.save_pretrained(exp_name)
        print(f"Training complete. Model saved to {exp_name}")
        if wandb.run:
            wandb.finish()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python train.py <path_to_config.yaml>")
        sys.exit(1)
    main(sys.argv[1])