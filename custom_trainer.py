from trl import GRPOTrainer
from transformers import PreTrainedModel
from accelerate import Accelerator
import torch
from typing import Union, Any, List, Dict
from accelerate.utils import gather
from trl.models import unwrap_model_for_generation
from trl.data_utils import is_conversational, maybe_apply_chat_template

class CustomGRPOTrainer(GRPOTrainer):
    def __init__(self, *args, **kwargs):
        # Extract custom parameters
        self.num_generations = kwargs.pop('num_generations', 16)  # Total completions per prompt
        self.prompts_per_batch = kwargs.pop('prompts_per_batch', 1)  # Unique prompts per accumulated batch
        num_gpus = self.accelerator.num_processes

        print(f"[DEBUG] Initializing CustomGRPOTrainer with num_generations={self.num_generations}, "
              f"prompts_per_batch={self.prompts_per_batch}, num_gpus={num_gpus}")

        # Compute generations per GPU
        if self.num_generations % num_gpus != 0:
            raise ValueError(f"num_generations ({self.num_generations}) must be divisible by num_gpus ({num_gpus})")
        self.generations_per_gpu = self.num_generations // num_gpus
        print(f"[DEBUG] Computed generations_per_gpu={self.generations_per_gpu}")

        # Compute total samples
        total_samples = self.prompts_per_batch * self.num_generations
        print(f"[DEBUG] Total samples per accumulated batch: {total_samples}")

        # Optimize batch distribution
        max_per_device = 16  # Adjust based on GPU memory
        per_device_train_batch_size = total_samples // num_gpus
        gradient_accumulation_steps = 1
        if per_device_train_batch_size > max_per_device:
            gradient_accumulation_steps = (total_samples + (num_gpus * max_per_device - 1)) // (num_gpus * max_per_device)
            per_device_train_batch_size = (total_samples + (gradient_accumulation_steps * num_gpus - 1)) // (gradient_accumulation_steps * num_gpus)
        print(f"[DEBUG] Computed per_device_train_batch_size={per_device_train_batch_size}, "
              f"gradient_accumulation_steps={gradient_accumulation_steps}")

        # Update training args
        self.args.per_device_train_batch_size = per_device_train_batch_size
        self.args.gradient_accumulation_steps = gradient_accumulation_steps

        # Validate effective batch size
        effective_batch_size = per_device_train_batch_size * num_gpus * gradient_accumulation_steps
        if effective_batch_size % self.num_generations != 0:
            raise ValueError(f"Effective batch size ({effective_batch_size}) must be divisible by num_generations ({self.num_generations})")
        print(f"[DEBUG] Validated effective_batch_size={effective_batch_size}")

        super().__init__(*args, **kwargs)
        print(f"[DEBUG] CustomGRPOTrainer initialized successfully")

    def _generate_and_score_completions(self, inputs: Dict[str, Union[torch.Tensor, Any]]) -> Dict[str, Union[torch.Tensor, Any]]:
        device = self.accelerator.device
        prompts = [x["prompt"] for x in inputs]
        prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs]

        print(f"[DEBUG] Processing prompts: len={len(prompts)}, type={type(prompts)}, sample={prompts[:2]}")
        print(f"[DEBUG] Prompts text: len={len(prompts_text)}, sample={prompts_text[:2]}")

        # Tokenize prompts
        prompt_inputs = self.processing_class(
            text=prompts_text, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False
        )
        prompt_inputs = super(GRPOTrainer, self)._prepare_inputs(prompt_inputs)
        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]

        print(f"[DEBUG] Tokenized prompt_ids shape: {prompt_ids.shape}, prompt_mask shape: {prompt_mask.shape}")

        # Truncate prompts if max_prompt_length is set
        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length:]
            prompt_mask = prompt_mask[:, -self.max_prompt_length:]
            print(f"[DEBUG] Truncated prompts to max_prompt_length={self.max_prompt_length}, new prompt_ids shape: {prompt_ids.shape}")

        # Generate completions using DTTModel
        with unwrap_model_for_generation(self.model_wrapped, self.accelerator) as unwrapped_model:
            generate_output = unwrapped_model.generate(
                prompt_ids,
                attention_mask=prompt_mask,
                max_new_tokens=self.max_completion_length,
                max_latent_steps=10,
                temperature=self.temperature,
                generations_per_prompt=self.generations_per_gpu,
            )

        prompt_completion_ids = generate_output['sequences']  # [batch_size * generations_per_gpu, max_len]
        total_latent_steps = generate_output['latent_steps']  # List of batch_size * generations_per_gpu ints

        print(f"[DEBUG] Generated prompt_completion_ids type={type(prompt_completion_ids)}, shape={prompt_completion_ids.shape}")
        print(f"[DEBUG] Total latent steps: len={len(total_latent_steps)}, sample={total_latent_steps[:5]}")
        assert isinstance(prompt_completion_ids, torch.Tensor), f"prompt_completion_ids is not a tensor, got {type(prompt_completion_ids)}"

        prompt_length = prompt_inputs["input_ids"].size(1)
        print(f"[DEBUG] Prompt length: {prompt_length}")
        if prompt_completion_ids.size(1) < prompt_length:
            raise ValueError(f"Generated sequences length {prompt_completion_ids.size(1)} is shorter than prompt_length {prompt_length}")

        # Slice the tensor into prompt and completion parts
        prompt_ids = prompt_completion_ids[:, :prompt_length]
        completion_ids = prompt_completion_ids[:, prompt_length:]

        print(f"[DEBUG] Split into prompt_ids shape={prompt_ids.shape}, completion_ids shape={completion_ids.shape}")

        # Compute completion mask based on EOS token
        is_eos = completion_ids == self.processing_class.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        print(f"[DEBUG] Computed completion_mask shape={completion_mask.shape}")

        # Concatenate prompt and completion masks
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        print(f"[DEBUG] Concatenated attention_mask shape={attention_mask.shape}")
        logits_to_keep = completion_ids.size(1)
        print(f"[DEBUG] Logits to keep: {logits_to_keep}")

        # Compute log probabilities
        with torch.no_grad():
            if self.num_iterations > 1:
                old_per_token_logps = self._get_per_token_logps(
                    self.model, prompt_completion_ids, attention_mask, logits_to_keep
                )
                print(f"[DEBUG] Computed old_per_token_logps shape={old_per_token_logps.shape if old_per_token_logps is not None else None}")
            else:
                old_per_token_logps = None
                print("[DEBUG] Skipped old_per_token_logps computation (num_iterations=1)")

            if self.beta == 0.0:
                ref_per_token_logps = None
                print("[DEBUG] Skipped ref_per_token_logps computation (beta=0.0)")
            elif self.ref_model is not None:
                ref_per_token_logps = self._get_per_token_logps(
                    self.ref_model, prompt_completion_ids, attention_mask, logits_to_keep
                )
                print(f"[DEBUG] Computed ref_per_token_logps shape={ref_per_token_logps.shape}")
            else:
                with self.accelerator.unwrap_model(self.model).disable_adapter():
                    ref_per_token_logps = self._get_per_token_logps(
                        self.model, prompt_completion_ids, attention_mask, logits_to_keep
                    )
                print(f"[DEBUG] Computed ref_per_token_logps (no ref_model) shape={ref_per_token_logps.shape}")

        # Prepare completions as token IDs (List[List[int]])
        completions = [seq.tolist() for seq in completion_ids]
        print(f"[DEBUG] Prepared completions: len={len(completions)}, sample_len={len(completions[0]) if completions else 0}")

        # Compute rewards using PhasedReward
        num_completions = len(prompts) * self.generations_per_gpu
        print(f"[DEBUG] Expected num_completions: {num_completions}")
        rewards_per_func = torch.zeros(num_completions, len(self.reward_funcs), device=device)
        for i, reward_func in enumerate(self.reward_funcs):
            rewards = reward_func(
                prompts=prompts,
                completions=completions,
                answer=[x["answer"] for x in inputs],
                latent_steps=total_latent_steps,
            )
            rewards_per_func[:, i] = torch.tensor(rewards, device=device)
            print(f"[DEBUG] Computed rewards for reward_func {i}: shape={rewards_per_func[:, i].shape}, sample={rewards_per_func[:5, i]}")

        # Gather rewards across processes
        rewards_per_func = gather(rewards_per_func)
        print(f"[DEBUG] Gathered rewards_per_func shape={rewards_per_func.shape}")
        rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).sum(dim=1)
        print(f"[DEBUG] Computed rewards shape={rewards.shape}, sample={rewards[:5]}")

        # Compute advantages with proper grouping
        global_batch_size = self.accelerator.num_processes * len(prompts)
        rewards_grouped = rewards.view(global_batch_size, self.num_generations)
        print(f"[DEBUG] Reshaped rewards_grouped shape={rewards_grouped.shape}")
        mean_grouped_rewards = rewards_grouped.mean(dim=1)
        std_grouped_rewards = rewards_grouped.std(dim=1)
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)
        print(f"[DEBUG] Computed advantages shape={advantages.shape}, sample={advantages[:5]}")

        # Slice for local process
        process_slice = slice(
            self.accelerator.process_index * num_completions,
            (self.accelerator.process_index + 1) * num_completions
        )
        advantages = advantages[process_slice]
        total_latent_steps = total_latent_steps[
            self.accelerator.process_index * num_completions:
            (self.accelerator.process_index + 1) * num_completions
        ]
        print(f"[DEBUG] Sliced advantages for local process: shape={advantages.shape}")
        print(f"[DEBUG] Sliced total_latent_steps: len={len(total_latent_steps)}")

        # Log metrics
        mode = "eval" if self.control.should_evaluate else "train"
        self._metrics[mode]["latent_steps"].append(sum(total_latent_steps) / len(total_latent_steps))
        print(f"[DEBUG] Logged latent_steps metric for mode={mode}: {self._metrics[mode]['latent_steps'][-1]}")

        return {
            "prompt_completion_ids": prompt_completion_ids,
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "old_per_token_logps": old_per_token_logps,
            "ref_per_token_logps": ref_per_token_logps,
            "advantages": advantages,
            "latent_steps": total_latent_steps
        }