from trl import GRPOTrainer
from transformers import PreTrainedModel
from accelerate import Accelerator
import torch
from typing import Union, Any, List, Dict
from accelerate.utils import gather
from trl.models import unwrap_model_for_generation
from trl.data_utils import is_conversational, maybe_apply_chat_template

class CustomGRPOTrainer(GRPOTrainer):
    def _generate_and_score_completions(self, inputs: Dict[str, Union[torch.Tensor, Any]]) -> Dict[str, Union[torch.Tensor, Any]]:
        """
        Generate completions and compute scores using the custom DTTModel and PhasedReward.

        Args:
            inputs (Dict[str, Union[torch.Tensor, Any]]): Input dictionary containing prompts and answers.

        Returns:
            Dict[str, Union[torch.Tensor, Any]]: Dictionary with generated IDs, masks, log probabilities, advantages, and latent steps.
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

        # Truncate prompts if max_prompt_length is set
        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length:]
            prompt_mask = prompt_mask[:, -self.max_prompt_length:]

        # Generate completions using DTTModel
        with unwrap_model_for_generation(self.model_wrapped, self.accelerator) as unwrapped_model:
            generate_output = unwrapped_model.generate(
                prompt_ids,
                attention_mask=prompt_mask,
                generation_config=self.generation_config,
                max_new_tokens=self.max_completion_length,
                max_latent_steps=50  # Custom parameter for DTTModel
            )

        # Extract sequences (tensor) and latent steps
        prompt_completion_ids = generate_output['sequences']  # Tensor of shape [batch_size * num_generations, seq_length]
        total_latent_steps = generate_output['latent_steps']  # List of latent steps per sequence

        # Debug prints to confirm tensor properties
        print(f"[DEBUG] Type of prompt_completion_ids: {type(prompt_completion_ids)}")
        assert isinstance(prompt_completion_ids, torch.Tensor), f"prompt_completion_ids is not a tensor, got {type(prompt_completion_ids)}"
        print(f"[DEBUG] Shape of prompt_completion_ids: {prompt_completion_ids.shape}")
        prompt_length = prompt_ids.size(1)
        print(f"[DEBUG] prompt_length: {prompt_length}")
        if prompt_completion_ids.size(1) < prompt_length:
            raise ValueError(f"Generated sequences length {prompt_completion_ids.size(1)} is shorter than prompt_length {prompt_length}")

        # Slice the tensor into prompt and completion parts
        prompt_ids = prompt_completion_ids[:, :prompt_length]
        completion_ids = prompt_completion_ids[:, prompt_length:]

        # Compute completion mask based on EOS token
        is_eos = completion_ids == self.processing_class.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        # Concatenate prompt and completion masks
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

        # Decode completions
        completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        completions = (
            [[{"role": "assistant", "content": (prompt.pop()["content"] if prompt[-1]["role"] == "assistant" else "") + c}]
            for prompt, c in zip(prompts, completions_text)]
            if is_conversational(inputs[0]) else completions_text
        )

        # Compute rewards using PhasedReward
        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
        for i, reward_func in enumerate(self.reward_funcs):
            # Assuming PhasedReward is the only reward function
            rewards = reward_func(
                prompts=prompts,
                completions=completions,
                answer=[x["answer"] for x in inputs],  # Ground truth answers from dataset
                latent_steps=total_latent_steps,
            )
            rewards_per_func[:, i] = torch.tensor(rewards, device=device)

        # Gather rewards across processes
        rewards_per_func = gather(rewards_per_func)
        rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).sum(dim=1)

        # Compute advantages
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

        # Log metrics
        mode = "eval" if self.control.should_evaluate else "train"
        self._metrics[mode]["latent_steps"].append(sum(total_latent_steps) / len(total_latent_steps))

        # Return dictionary with all required tensors
        return {
            "prompt_completion_ids": prompt_completion_ids,  # Tensor for base _prepare_inputs slicing
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "old_per_token_logps": old_per_token_logps,
            "ref_per_token_logps": ref_per_token_logps,
            "advantages": advantages,
            "latent_steps": total_latent_steps
        }