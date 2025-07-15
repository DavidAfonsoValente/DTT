import os
import sys
import yaml
import torch
import wandb
import transformers
from dataclasses import dataclass, field

# Import the Accelerator class
from accelerate import Accelerator
from transformers import GenerationConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer

from model import SparseGatedModel
from reward import custom_reward
from utils import preprocess_dataset

from torch import nn
import torch.nn.functional as F
from typing import Any, Dict, List, Optional, Tuple, Union

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

    def _prepare_inputs(self, inputs: Dict[str, Union[torch.Tensor, Any]]) -> Dict[str, Union[torch.Tensor, Any]]:
        """
        Overrides the GRPOTrainer's data preparation to prevent it from
        calling the reward function, which is handled within our custom training_step.
        This implementation simply moves data to the correct device.
        """
        # By calling super(GRPOTrainer, self), we explicitly call the
        # _prepare_inputs method from the grandparent class (transformers.Trainer),
        # skipping the problematic GRPOTrainer implementation.
        return transformers.Trainer._prepare_inputs(self, inputs)
    
    def _anneal_gumbel_temperature(self):
        if self.total_steps_for_annealing <= 0: return self.args.initial_gate_temperature
        progress = min(1.0, self.state.global_step / self.total_steps_for_annealing)
        annealed_temp = self.args.initial_gate_temperature - \
                        (self.args.initial_gate_temperature - self.args.min_gate_temperature) * progress
        # The model is wrapped by DDP, access the underlying module
        self.model.module.gate_temperature = max(self.args.min_gate_temperature, annealed_temp)
        return self.model.module.gate_temperature

    @torch.no_grad()
    def _custom_generate_sequences(self, input_ids, attention_mask):
        # The model is wrapped by DDP, access the underlying module for eval and custom attrs
        self.model.module.eval()
        batch_size, prompt_len = input_ids.shape
        device = self.model.module.device
        
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
            return_last_hidden_state=True
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
                return_last_hidden_state=True
            )
            
            step_gate_values = self.model.module.current_gate_values_for_batch.squeeze()
            for i in range(batch_size):
                all_gate_values[i].append(step_gate_values[i].item() if batch_size > 1 else step_gate_values.item())

            past_key_values = lm_outputs_step.past_key_values
            prev_h_for_next_step = h_t_current_step
            next_token_logits = lm_outputs_step.logits[:, -1, :]
        
        self.model.module.train()
        return generated_ids, all_gate_values, current_gumbel_model_temp

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], num_items_in_batch: int) -> torch.Tensor:
        """
        Override training_step to implement the custom generation and reward pipeline.
        """
        # Move original batch to device
        inputs = self._prepare_inputs(inputs)
        
        # 1. Generate completions and get gate values using our custom method
        prompt_input_ids = inputs["input_ids"]
        prompt_attention_mask = inputs["attention_mask"]
        prompt_len = prompt_input_ids.shape[1]

        expanded_input_ids = prompt_input_ids.repeat_interleave(self.args.num_generations, dim=0)
        expanded_attention_mask = prompt_attention_mask.repeat_interleave(self.args.num_generations, dim=0)

        with torch.no_grad():
            sequences_ids, gate_values_list, current_gumbel_temp = self._custom_generate_sequences(
                expanded_input_ids, expanded_attention_mask
            )
        
        # 2. Compute rewards using our custom reward function
        completions_ids = [s.tolist() for s in sequences_ids]
        rewards = self.reward_funcs[0](
            completions_ids=completions_ids,
            gate_values_list=gate_values_list,
            tokenizer=self.processing_class,
            ground_truths=inputs["ground_truths"],
            prompt_len=prompt_len,
            lambda_penalty=self.args.lambda_penalty,
            gate_penalty_coeff=self.args.gate_penalty_coeff,
            num_generations=self.args.num_generations,
        )
        
        # 3. Form chosen/rejected pairs based on rewards
        rewards_tensor = torch.tensor(rewards, device=self.accelerator.device).view(-1, self.args.num_generations)
        chosen_indices = torch.argmax(rewards_tensor, dim=-1)
        
        # Create pairs of (chosen, rejected)
        chosen_responses = []
        rejected_responses = []
        
        for i in range(len(prompt_input_ids)):
            chosen_idx = chosen_indices[i]
            # global index in the flattened batch
            global_chosen_idx = i * self.args.num_generations + chosen_idx
            
            for j in range(self.args.num_generations):
                if i * self.args.num_generations + j != global_chosen_idx:
                    chosen_responses.append(sequences_ids[global_chosen_idx])
                    rejected_responses.append(sequences_ids[i * self.args.num_generations + j])

        # 4. Get log probabilities for the chosen and rejected sequences
        # This requires a forward pass with the model in training mode
        model.train()
        
        chosen_toks = self._left_pad_sequences(chosen_responses, self.processing_class.pad_token_id)
        rejected_toks = self._left_pad_sequences(rejected_responses, self.processing_class.pad_token_id)

        all_toks = torch.cat((chosen_toks, rejected_toks), dim=0)
        all_logps, _ = self.get_logps(model, all_toks, prompt_len)
        
        chosen_logps = all_logps[:len(chosen_responses)]
        rejected_logps = all_logps[len(chosen_responses):]
        
        # 5. Compute GRPO loss
        loss = -F.logsigmoid(self.beta * (chosen_logps - rejected_logps)).mean()
        
        # Logging
        if self.accelerator.is_main_process and self.state.global_step % self.args.logging_steps == 0:
            metrics = self._calculate_and_log_metrics(
                completions_ids, gate_values_list, rewards, inputs["ground_truths"], prompt_len, current_gumbel_temp
            )
            self._log_to_progress_table(self.state.global_step, metrics, loss)

        return loss

    def _calculate_and_log_metrics(self, completions, gates, rewards, gts, p_len, gumbel_temp):
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
                g_pens.append(self.args.gate_penalty_coeff * (g_pen_sum / len(gen_gates)) if gen_gates else 0.0)
            else:
                g_pens.append(0.0)

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
        if self.accelerator.is_main_process:
            self.progress_table.add_data(
                step, loss.item() if loss else float('nan'), metrics.get("reward_mean", 0), metrics.get("correctness_mean", 0),
                metrics.get("token_penalty_mean", 0), metrics.get("gate_penalty_mean", 0), metrics.get("gate_mean_overall", 0),
                metrics.get("gate_mean_reasoning", 0), metrics.get("gate_mean_non_reasoning", 0), metrics.get("gumbel_temp_model", 0)
            )
            if step > 0 and step % (self.args.logging_steps * 10) == 0:
                wandb.log({"train_progress_summary": self.progress_table})

    def _left_pad_sequences(self, sequences, pad_value):
        """Left pads a list of tensors to the same length."""
        max_len = max(len(seq) for seq in sequences)
        padded_sequences = [
            torch.cat([torch.full((max_len - len(seq),), pad_value, dtype=seq.dtype, device=seq.device), seq]) 
            for seq in sequences
        ]
        return torch.stack(padded_sequences)

def main(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Instantiate the Accelerator at the beginning of main
    accelerator = Accelerator()

    exp_name = config["output_dir"]
    # Use the accelerator's check for the main process
    if accelerator.is_main_process:
        if os.path.exists(exp_name) and os.listdir(exp_name) and not config.get("overwrite_output_dir"):
            print(f"Directory {exp_name} exists and is not empty. Exiting.")
            return
        wandb.init(project=config.get("wandb_project", "dtt-project"), name=exp_name, config=config)
        wandb.save(config_path)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    # Use the explicit device_map to load one full model per process/GPU
    base_model = AutoModelForCausalLM.from_pretrained(
        config["model_name"],
        quantization_config=bnb_config,
        device_map={"": accelerator.device}
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
        report_to="wandb" if accelerator.is_main_process else None,
        remove_unused_columns=False,
        bf16=config.get("use_bf16", False),
        fp16=config.get("use_fp16", False),
        
        beta=config["beta_grpo"],
        temperature=config["sampling_temperature_grpo"],
        num_generations=config["group_size_grpo"],
        max_prompt_length=config["max_prompt_length"],
        max_completion_length=config["max_completion_length"],
        
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

    if accelerator.is_main_process:
        # It's good practice to unwrap the model before saving
        unwrapped_model = accelerator.unwrap_model(trainer.model)
        unwrapped_model.save_pretrained(exp_name)
        print(f"Training complete. Model saved to {exp_name}")
        if wandb.run:
            wandb.finish()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python train.py <path_to_config.yaml>")
        sys.exit(1)
    main(sys.argv[1])