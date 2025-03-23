import torch
from typing import List
from torch.nn.functional import cosine_similarity, softmax

class PhasedReward:
    def __init__(
        self,
        model,
        total_steps: int,
        tokenizer,
        gamma: float = 0.98,
        lambda_eff: float = 0.1,
        G: int = 8,
        enable_binary: bool = True,
        enable_crs: bool = True,
        enable_lcr: bool = True,
        enable_ede: bool = True,
        enable_eff: bool = True,
    ):
        """
        Initialize the PhasedReward class with access to the model and toggleable reward components.

        Args:
            model: The DTTModel instance to access hidden states and logits.
            total_steps (int): Total number of training steps.
            tokenizer: Tokenizer used to decode completions.
            gamma (float): Discount factor for efficiency reward.
            lambda_eff (float): Weight for efficiency reward.
            G (int): Number of generations per prompt.
            enable_binary (bool): Flag to enable/disable binary reward.
            enable_crs (bool): Flag to enable/disable CRS reward.
            enable_lcr (bool): Flag to enable/disable LCR reward.
            enable_ede (bool): Flag to enable/disable EDE reward.
            enable_eff (bool): Flag to enable/disable efficiency reward.
        """
        self.model = model
        self.total_steps = total_steps
        self.tokenizer = tokenizer
        self.gamma = gamma
        self.lambda_eff = lambda_eff
        self.G = G
        self.enable_binary = enable_binary
        self.enable_crs = enable_crs
        self.enable_lcr = enable_lcr
        self.enable_ede = enable_ede
        self.enable_eff = enable_eff
        self.current_step = 0

    def __call__(
        self,
        prompts: List[List[int]],
        completions: List[List[int]],
        answer: List[str],
        latent_steps: List[int],
        **kwargs,
    ) -> List[float]:
        """
        Compute the total rewards for the completions based on the current training phase and enabled components.

        Args:
            prompts (List[List[int]]): List of prompt token IDs (size B).
            completions (List[List[int]]): List of generated completion token IDs (size B*G).
            answer (List[str]): List of ground truth answers (size B).
            latent_steps (List[int]): List of total latent steps for each completion (size B*G).
            **kwargs: Additional keyword arguments.

        Returns:
            List[float]: List of total rewards for each completion (size B*G).
        """
        # Determine the current phase and set default weights
        progress = self.current_step / self.total_steps
        if progress <= 0.2:  # Warmup Phase (0-20%)
            phase = "Warmup"
            w_binary_default = 0.7
            w_crs_default = 0.2
            w_lcr_default = 0.1
            w_ede_default = 0.0
        elif progress <= 0.9:  # Core Phase (20-90%)
            phase = "Core"
            w_binary_default = 0.8
            w_crs_default = 0.15
            w_lcr_default = 0.0
            w_ede_default = 0.05
        else:  # Final Phase (90-100%)
            phase = "Final"
            w_binary_default = 1.0
            w_crs_default = 0.0
            w_lcr_default = 0.0
            w_ede_default = 0.0

        # Adjust weights based on enabled flags
        w_binary = w_binary_default if self.enable_binary else 0.0
        w_crs = w_crs_default if self.enable_crs else 0.0
        w_lcr = w_lcr_default if self.enable_lcr else 0.0
        w_ede = w_ede_default if self.enable_ede else 0.0

        # Normalize weights of enabled components (excluding efficiency)
        sum_weights = w_binary + w_crs + w_lcr + w_ede
        if sum_weights > 0:
            w_binary /= sum_weights
            w_crs /= sum_weights
            w_lcr /= sum_weights
            w_ede /= sum_weights

        B = len(prompts)
        total_rewards = []

        for i in range(B):
            group_completions = completions[i * self.G : (i + 1) * self.G]
            latent_steps_group = latent_steps[i * self.G : (i + 1) * self.G]
            actual_answer = answer[i]

            # Compute binary rewards if needed for binary or CRS
            if self.enable_binary or self.enable_crs:
                binary_rewards = [
                    self.compute_binary_reward(comp, actual_answer)
                    for comp in group_completions
                ]
            else:
                binary_rewards = [0.0] * self.G

            # Compute CRS rewards if enabled
            if self.enable_crs:
                r_crs = self.compute_crs(binary_rewards)
            else:
                r_crs = [0.0] * self.G

            # Compute efficiency rewards if enabled
            if self.enable_eff:
                r_eff = [
                    self.gamma ** ls
                    for ls in latent_steps_group
                ]
            else:
                r_eff = [0.0] * self.G

            # Compute LCR rewards if enabled
            if self.enable_lcr:
                r_lcr = [
                    self.compute_lcr(self.model.last_hidden_states[j])
                    for j in range(i * self.G, (i + 1) * self.G)
                ]
            else:
                r_lcr = [0.0] * self.G

            # Compute EDE rewards if enabled
            if self.enable_ede:
                r_ede = [
                    self.compute_ede(self.model.last_logits[j], progress)
                    for j in range(i * self.G, (i + 1) * self.G)
                ]
            else:
                r_ede = [0.0] * self.G

            # Combine rewards with adjusted weights
            r_total_group = [
                (w_binary * br + w_crs * crs + w_lcr * lcr + w_ede * ede) +
                (self.lambda_eff * eff if self.enable_eff else 0)
                for br, crs, lcr, ede, eff in zip(binary_rewards, r_crs, r_lcr, r_ede, r_eff)
            ]

            total_rewards.extend(r_total_group)

        self.current_step += 1
        return total_rewards

    def compute_binary_reward(self, completion: List[int], actual_answer: str) -> float:
        """
        Compute the binary reward by comparing the completion to the actual answer.

        Args:
            completion (List[int]): Token IDs of the completion.
            actual_answer (str): Ground truth answer.

        Returns:
            float: 1.0 if correct, 0.0 otherwise.
        """
        text = self.tokenizer.decode(completion, skip_special_tokens=False)
        if "###" in text:
            answer_part = text.split("###")[-1].strip()
            clean_answer = answer_part.replace(",", "").strip()
        else:
            clean_answer = text.strip()
        return 1.0 if clean_answer == actual_answer else 0.0

    def compute_crs(self, binary_rewards: List[float]) -> List[float]:
        """
        Compute Contrastive Reward Shaping (CRS) rewards for the group.

        Args:
            binary_rewards (List[float]): Binary rewards for the group.

        Returns:
            List[float]: CRS rewards for each completion.
        """
        br_tensor = torch.tensor(binary_rewards, dtype=torch.float32)
        softmax_br = torch.softmax(br_tensor, dim=0)
        r_crs = (softmax_br - 0.5).tolist()  # Center around 0
        return r_crs

    def compute_lcr(self, hidden_states: List[torch.Tensor]) -> float:
        """
        Compute Latent Consistency Regularization (LCR) reward.

        Args:
            hidden_states (List[torch.Tensor]): Hidden states during latent reasoning.

        Returns:
            float: Average cosine similarity between consecutive hidden states.
        """
        if len(hidden_states) < 2:  # Need at least two states for similarity
            return 0.0
        similarities = [
            cosine_similarity(h1, h2, dim=0).item()
            for h1, h2 in zip(hidden_states[:-1], hidden_states[1:])
        ]
        return sum(similarities) / len(similarities) if similarities else 0.0

    def compute_ede(self, logits: List[torch.Tensor], progress: float) -> float:
        """
        Compute Entropy-Driven Exploration (EDE) reward.

        Args:
            logits (List[torch.Tensor]): Logits for each generated token.
            progress (float): Current training progress (step/total_steps).

        Returns:
            float: Scaled average entropy of the policy distribution.
        """
        if not logits:
            return 0.0
        entropies = [
            -(softmax(logit, dim=-1) * torch.log_softmax(logit, dim=-1)).sum(dim=-1).mean().item()
            for logit in logits
        ]
        avg_entropy = sum(entropies) / len(entropies) if entropies else 0.0
        scale = 0.2 * (1 - progress)  # Scale decreases over training
        return scale * avg_entropy