# reward.py
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
        progress = self.current_step / self.total_steps
        if progress <= 0.2:
            phase = "Warmup"
            w_binary_default = 0.7
            w_crs_default = 0.2
            w_lcr_default = 0.1
            w_ede_default = 0.0
        elif progress <= 0.9:
            phase = "Core"
            w_binary_default = 0.8
            w_crs_default = 0.15
            w_lcr_default = 0.0
            w_ede_default = 0.05
        else:
            phase = "Final"
            w_binary_default = 1.0
            w_crs_default = 0.0
            w_lcr_default = 0.0
            w_ede_default = 0.0

        w_binary = w_binary_default if self.enable_binary else 0.0
        w_crs = w_crs_default if self.enable_crs else 0.0
        w_lcr = w_lcr_default if self.enable_lcr else 0.0
        w_ede = w_ede_default if self.enable_ede else 0.0

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

            if self.enable_binary or self.enable_crs:
                binary_rewards = [
                    self.compute_binary_reward(comp, actual_answer)
                    for comp in group_completions
                ]
            else:
                binary_rewards = [0.0] * self.G

            if self.enable_crs:
                r_crs = self.compute_crs(binary_rewards)
            else:
                r_crs = [0.0] * self.G

            if self.enable_eff:
                r_eff = [
                    self.gamma ** ls
                    for ls in latent_steps_group
                ]
            else:
                r_eff = [0.0] * self.G

            if self.enable_lcr:
                r_lcr = [
                    self.compute_lcr(self.model.last_hidden_states[j])
                    for j in range(i * self.G, (i + 1) * self.G)
                ]
            else:
                r_lcr = [0.0] * self.G

            if self.enable_ede:
                r_ede = [
                    self.compute_ede(self.model.last_logits[j], progress)
                    for j in range(i * self.G, (i + 1) * self.G)
                ]
            else:
                r_ede = [0.0] * self.G

            r_total_group = [
                (w_binary * br + w_crs * crs + w_lcr * lcr + w_ede * ede) +
                (self.lambda_eff * eff if self.enable_eff else 0)
                for br, crs, lcr, ede, eff in zip(binary_rewards, r_crs, r_lcr, r_ede, r_eff)
            ]

            total_rewards.extend(r_total_group)

        self.current_step += 1
        return total_rewards

    def compute_binary_reward(self, completion: List[int], actual_answer: str) -> float:
        text = self.tokenizer.decode(completion, skip_special_tokens=False)
        if "###" in text:
            answer_part = text.split("###")[-1].strip()
            clean_answer = answer_part.replace(",", "").strip()
        else:
            clean_answer = text.strip()
        return 1.0 if clean_answer == actual_answer else 0.0

    def compute_crs(self, binary_rewards: List[float]) -> List[float]:
        br_tensor = torch.tensor(binary_rewards, dtype=torch.float32)
        softmax_br = torch.softmax(br_tensor, dim=0)
        r_crs = (softmax_br - 0.5).tolist()
        return r_crs

    def compute_lcr(self, hidden_states: List[torch.Tensor]) -> float:
        if len(hidden_states) < 2:
            return 0.0
        similarities = [
            cosine_similarity(h1, h2, dim=0).item()
            for h1, h2 in zip(hidden_states[:-1], hidden_states[1:])
        ]
        return sum(similarities) / len(similarities) if similarities else 0.0

    def compute_ede(self, logits: List[torch.Tensor], progress: float) -> float:
        if not logits:
            return 0.0
        entropies = [
            -(softmax(logit, dim=-1) * torch.log_softmax(logit, dim=-1)).sum(dim=-1).mean().item()
            for logit in logits
        ]
        avg_entropy = sum(entropies) / len(entropies) if entropies else 0.0
        scale = 0.2 * (1 - progress)
        return scale * avg_entropy