from trl import GRPOTrainer
from transformers import PreTrainedModel
from accelerate import Accelerator
import torch
from typing import Union, Any, List, Dict
from accelerate.utils import gather_object
from trl.models import unwrap_model_for_generation
from trl.data_utils import maybe_apply_chat_template

class CustomGRPOTrainer(GRPOTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_generations = kwargs.get('args').num_generations  # Total completions per prompt
        self.generations_per_gpu = self.num_generations // self.accelerator.num_processes
        print(f"[DEBUG] Initialized CustomGRPOTrainer: num_generations={self.num_generations}, generations_per_gpu={self.generations_per_gpu}", flush=True)

    def _generate_and_score_completions(self, inputs: Dict[str, Union[torch.Tensor, Any]]) -> Dict[str, Union[torch.Tensor, Any]]:
        device = self.accelerator.device
        prompts = [x["prompt"] for x in inputs]
        prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs]

        print(f"[DEBUG] Type of prompts: {type(prompts)}", flush=True)
        print(f"[DEBUG] Prompts: {prompts}", flush=True)

        # Tokenize prompts
        prompt_inputs = self.processing_class(
            text=prompts_text, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False
        )
        prompt_inputs = super(GRPOTrainer, self)._prepare_inputs(prompt_inputs)
        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]

        # Truncate prompts if max_prompt_length is set
        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length:]
            prompt_mask = prompt_mask[:, -self.max_prompt_length:]

        # Generate completions
        with unwrap_model_for_generation(self.model_wrapped, self.accelerator) as unwrapped_model:
            generate_output = unwrapped_model.generate(
                input_ids=prompt_ids,
                attention_mask=prompt_mask,
                max_new_tokens=self.max_completion_length,
                max_latent_steps=10,
                temperature=self.temperature,
                generations_per_prompt=self.generations_per_gpu,
            )

        prompt_completion_ids = generate_output['sequences']  # [batch_size * generations_per_gpu, max_len]
        total_latent_steps = generate_output['latent_steps']

        print(f"[DEBUG] Shape of prompt_completion_ids: {prompt_completion_ids.shape}", flush=True)
        prompt_length = prompt_inputs["input_ids"].size(1)
        if prompt_completion_ids.size(1) < prompt_length:
            raise ValueError(f"Generated sequences length {prompt_completion_ids.size(1)} is shorter than prompt_length {prompt_length}")

        # Slice into prompt and completion parts
        prompt_ids = prompt_completion_ids[:, :prompt_length]
        completion_ids = prompt_completion_ids[:, prompt_length:]

        # Compute completion mask
        is_eos = completion_ids == self.processing_class.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        # Concatenate masks
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)

        # Compute log probabilities
        with torch.no_grad():
            if self.num_iterations > 1:
                old_per_token_logps = self._get_per_token_logps(
                    self.model, prompt_completion_ids, attention_mask, logits_to_keep
                )
            else:
                old_per_token_logps = None

            if self.beta == 0.0:
                ref_per_token_logps = None
            elif self.ref_model is not None:
                ref_per_token_logps = self._get_per_token_logps(
                    self.ref_model, prompt_completion_ids, attention_mask, logits_to_keep
                )
            else:
                with self.accelerator.unwrap_model(self.model).disable_adapter():
                    ref_per_token_logps = self._get_per_token_logps(
                        self.model, prompt_completion_ids, attention_mask, logits_to_keep
                    )

        # Prepare completions for reward computation
        completions = [seq.tolist() for seq in completion_ids]

        # Compute rewards
        num_completions = len(prompts) * self.generations_per_gpu
        rewards_per_func = torch.zeros(num_completions, len(self.reward_funcs), device=device)
        for i, reward_func in enumerate(self.reward_funcs):
            rewards = reward_func(
                prompts=prompts,
                completions=completions,
                answer=[x["answer"] for x in inputs],
                latent_steps=total_latent_steps,
            )
            rewards_per_func[:, i] = torch.tensor(rewards, device=device)

        # Gather rewards across processes
        rewards_per_func = gather_object(rewards_per_func)
        rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).sum(dim=1)

        # Compute advantages
        global_batch_size = self.accelerator.num_processes * len(prompts)
        rewards_grouped = rewards.view(global_batch_size, self.num_generations)
        mean_grouped_rewards = rewards_grouped.mean(dim=1)
        std_grouped_rewards = rewards_grouped.std(dim=1)
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)

        # Slice for local process
        process_slice = slice(
            self.accelerator.process_index * num_completions,
            (self.accelerator.process_index + 1) * num_completions
        )
        advantages = advantages[process_slice]
        total_latent_steps = total_latent_steps[process_slice.start:process_slice.stop]

        # Log metrics
        mode = "eval" if self.control.should_evaluate else "train"
        self._metrics[mode]["latent_steps"].append(sum(total_latent_steps) / len(total_latent_steps))

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