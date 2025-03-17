from trl import GRPOTrainer
from transformers import PreTrainedModel
from accelerate import Accelerator
import torch
from typing import Union, Any, List, Dict  # Added import for type hints

class CustomGRPOTrainer(GRPOTrainer):
    def _generate_and_score_completions(self, inputs: Dict[str, Union[torch.Tensor, Any]]) -> Dict[str, Union[torch.Tensor, Any]]:
        """
        Generate completions and compute scores, handling the DTTModel's dictionary output with sequences and latent steps.

        Args:
            inputs (dict): Input dictionary containing 'prompt' and other dataset columns.

        Returns:
            dict: Processed inputs with prompt IDs, masks, completion IDs, masks, log probabilities, advantages, and latent steps.
        """
        device = self.accelerator.device
        prompts = [x["prompt"] for x in inputs]
        prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs]
        
        # Tokenize prompts
        prompt_inputs = self.processing_class(
            text=prompts_text, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False
        )
        prompt_inputs = super()._prepare_inputs(prompt_inputs)
        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]

        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length:]
            prompt_mask = prompt_mask[:, -self.max_prompt_length:]

        # Generate completions using the model's generate method
        with unwrap_model_for_generation(self.model_wrapped, self.accelerator) as unwrapped_model:
            generate_output = unwrapped_model.generate(
                prompt_ids,
                attention_mask=prompt_mask,
                generation_config=self.generation_config,
                max_new_tokens=self.max_completion_length,
                max_latent_steps=50  # Adjust as per your DTTModel configuration
            )

        # Extract sequences and latent steps from the generate output
        prompt_completion_ids = generate_output['sequences']
        total_latent_steps = generate_output['latent_steps']

        # Compute prompt length and extract completion IDs
        prompt_length = prompt_ids.size(1)
        prompt_ids = prompt_completion_ids[:, :prompt_length]
        completion_ids = prompt_completion_ids[:, prompt_length:]

        # Mask everything after the first EOS token
        is_eos = completion_ids == self.processing_class.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        # Concatenate masks for logit computation
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)

        logits_to_keep = completion_ids.size(1)

        with torch.no_grad():
            # Compute old log probabilities if multi-iteration
            if self.num_iterations > 1:
                old_per_token_logps = self._get_per_token_logps(
                    self.model, prompt_completion_ids, attention_mask, logits_to_keep
                )
            else:
                old_per_token_logps = None

            # Compute reference log probabilities if beta > 0
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

        # Decode completions
        completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        completions = (
            [[{"role": "assistant", "content": (prompt.pop()["content"] if prompt[-1]["role"] == "assistant" else "") + c}]
             for prompt, c in zip(prompts, completions_text)]
            if is_conversational(inputs[0]) else completions_text
        )

        # Compute rewards, passing latent_steps directly
        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
        for i, (reward_func, reward_processing_class) in enumerate(zip(self.reward_funcs, self.reward_processing_classes)):
            reward_func_name = (
                f"reward {reward_func.config._name_or_path.split('/')[-1]}"
                if isinstance(reward_func, PreTrainedModel) else reward_func.__name__
            )
            with profiling_context(self, reward_func_name):
                if isinstance(reward_func, PreTrainedModel):
                    # Handle model-based reward
                    texts = (
                        [apply_chat_template({"messages": p + c}, reward_processing_class)["text"] for p, c in zip(prompts, completions)]
                        if is_conversational(inputs[0]) else [p + c for p, c in zip(prompts, completions)]
                    )
                    reward_inputs = reward_processing_class(
                        text=texts, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False
                    )
                    reward_inputs = super()._prepare_inputs(reward_inputs)
                    with torch.inference_mode():
                        rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]
                else:
                    # Custom reward function with latent_steps
                    reward_kwargs = {key: [example[key] for example in inputs] for key in inputs[0] if key not in ["prompt", "completion"]}
                    output_reward_func = reward_func(
                        prompts=prompts,
                        completions=completions,
                        latent_steps=total_latent_steps,
                        **reward_kwargs
                    )
                    rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

        # Gather and normalize rewards
        rewards_per_func = gather(rewards_per_func)
        rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).sum(dim=1)
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)

        # Slice for local process
        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts)
        )
        advantages = advantages[process_slice]
        total_latent_steps = total_latent_steps[self.accelerator.process_index * len(prompts):(self.accelerator.process_index + 1) * len(prompts)]

        # Log metrics (optional expansion here if needed)
        mode = "eval" if self.control.should_evaluate else "train"
        self._metrics[mode]["latent_steps"].append(sum(total_latent_steps) / len(total_latent_steps))

        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "old_per_token_logps": old_per_token_logps,
            "ref_per_token_logps": ref_per_token_logps,
            "advantages": advantages,
            "latent_steps": total_latent_steps  # Pass latent steps for potential use in compute_loss if needed
        }