import torch
from typing import List

class PhasedReward:
    def __init__(
        self,
        total_steps: int,
        tokenizer,
        gamma: float = 0.98,
        lambda_eff: float = 0.1,
        G: int = 8,
    ):
        """
        Initialize the PhasedReward class.

        Args:
            total_steps (int): Total number of training steps.
            tokenizer: Tokenizer used to decode completions.
            gamma (float): Discount factor for efficiency reward.
            lambda_eff (float): Weight for efficiency reward.
            G (int): Number of generations per prompt.
        """
        self.total_steps = total_steps
        self.tokenizer = tokenizer
        self.gamma = gamma
        self.lambda_eff = lambda_eff
        self.G = G
        self.current_step = 0

    def __call__(
        self,
        prompts: List[List[int]],
        completions: List[List[int]],
        answer: List[str],
        **kwargs,
    ) -> List[float]:
        """
        Compute the total rewards for the completions based on the current training phase.

        Args:
            prompts (List[List[int]]): List of prompt token IDs (size B).
            completions (List[List[int]]): List of generated completion token IDs (size B*G).
            answer (List[str]): List of ground truth answers (size B).
            **kwargs: Additional keyword arguments (e.g., other dataset columns).

        Returns:
            List[float]: List of total rewards for each completion (size B*G).
        """
        # Determine the current phase based on training progress
        progress = self.current_step / self.total_steps
        if progress <= 0.2:
            phase = "Warmup"
            w_binary = 0.7
            w_crs = 0.2
        elif progress <= 0.9:
            phase = "Core"
            w_binary = 0.8
            w_crs = 0.15
        else:
            phase = "Final"
            w_binary = 1.0
            w_crs = 0.0

        # Number of prompts in the batch
        B = len(prompts)
        total_rewards = []

        # Group completions into [completions[i*G:(i+1)*G] for i in range(B)]
        for i in range(B):
            group_completions = completions[i * self.G : (i + 1) * self.G]
            actual_answer = answer[i]

            # Compute binary rewards for the group
            binary_rewards = [
                self.compute_binary_reward(comp, actual_answer)
                for comp in group_completions
            ]

            # Compute CRS rewards
            r_crs = self.compute_crs(binary_rewards)

            # Compute efficiency rewards
            r_eff = [
                self.gamma ** self.compute_n_lt(comp) for comp in group_completions
            ]

            # Combine rewards based on phase weights
            r_total_group = [
                w_binary * br + w_crs * crs + self.lambda_eff * eff
                for br, crs, eff in zip(binary_rewards, r_crs, r_eff)
            ]

            # Append to total rewards
            total_rewards.extend(r_total_group)

        # Increment the current step
        self.current_step += 1

        return total_rewards

    def compute_binary_reward(self, completion: List[int], actual_answer: str) -> float:
        """
        Compute the binary reward by comparing the extracted answer from the completion
        with the actual answer.

        Args:
            completion (List[int]): Token IDs of the completion.
            actual_answer (str): Ground truth answer.

        Returns:
            float: 1.0 if the extracted answer matches the actual answer, 0.0 otherwise.
        """
        text = self.tokenizer.decode(completion, skip_special_tokens=False)
        if "###" in text:
            answer_part = text.split("###")[-1].strip()
            clean_answer = answer_part.replace(",", "").strip()
        else:
            clean_answer = text.strip()
        return 1.0 if clean_answer == actual_answer else 0.0

    def compute_n_lt(self, completion: List[int]) -> int:
        """
        Compute the total number of latent steps in the completion.

        Args:
            completion (List[int]): Token IDs of the completion.

        Returns:
            int: Total number of latent steps.
        """
        start_id = self.tokenizer.convert_tokens_to_ids("<|start-latent|>")
        end_id = self.tokenizer.convert_tokens_to_ids("<|end-latent|>")
        n_lt = 0
        i = 0
        while i < len(completion):
            if completion[i] == start_id:
                j = i + 1
                while j < len(completion) and completion[j] != end_id:
                    j += 1
                if j < len(completion):
                    n_lt += j - i - 1  # Tokens between start and end, exclusive
                    i = j + 1
                else:
                    break
            else:
                i += 1
        return n_lt

    def compute_crs(self, binary_rewards: List[float]) -> List[float]:
        """
        Compute the Contrastive Reward Shaping (CRS) rewards for the group.

        Args:
            binary_rewards (List[float]): Binary rewards for the group.

        Returns:
            List[float]: CRS rewards for each completion in the group.
        """
        br_tensor = torch.tensor(binary_rewards)
        softmax_br = torch.softmax(br_tensor, dim=0)
        r_crs = softmax_br - 0.5
        return r_crs.tolist()