Dynamic Thinking Tokens (DTT) Project Description
Introduction
The Dynamic Thinking Tokens (DTT) project trains a GPT-2-based language model to generate structured outputs for reasoning tasks, producing sequences in the format [bot]...reasoning...[eot]...answer. It employs a Gumbel-sigmoid gating mechanism to decide when to use hidden states for latent reasoning (gate ≈ 1) or token embeddings (gate ≈ 0), inspired by the Chain of Continuous Thought (COCONUT) method (arXiv:2412.06769) and related works on hybrid reasoning. The project supports datasets like GSM8K, ProsQA, and ProntoQA, uses Group Relative Policy Optimization (GRPO) for training, and is optimized for distributed training on 4 GPUs using Hugging Face's Accelerate library.
Background
COCONUT: Chain of Continuous Thought
COCONUT, described in "Training Large Language Models to Reason in a Continuous Latent Space" (arXiv:2412.06769), enables LLMs to reason in a continuous latent space by feeding the last hidden state back as the next input embedding. This avoids generating intermediate text tokens, improving efficiency for tasks like mathematical reasoning or logical puzzles that require backtracking. COCONUT’s approach is particularly effective for complex reasoning, as it reduces the overhead of textual coherence.
Project Description
The DTT project builds on COCONUT’s latent efficiency to create a model that generates structured outputs with minimal reasoning tokens. The model uses a Gumbel-sigmoid gating mechanism to control hidden state usage, trained with GRPO to optimize correctness, efficiency, and gate sparsity.
Model Architecture

DTTModel: Extends Hugging Face’s GPT2LMHeadModel for GPT-2 small, with:
Gate Linear Layer: Produces gate logits from hidden states.
Gumbel-Sigmoid: Ensures near-binary gate values for sparsity, controlled by a temperature parameter.


Forward Pass: Combines token embeddings and hidden states based on gate values, with attention masking during bootstrap.
Generate Method: Custom generation loop that dynamically builds input embeddings for blended latent reasoning.

Training Process

Two-Stage Training:
Stage 1 (Bootstrap): Supervised fine-tuning with synthetic data (15% of samples include random fillers between [bot] and [eot]). Uses cross-entropy loss plus hinge regularizer on gates, with attention scaled (0.3) in noisy spans.
Stage 2 (GRPO): Reinforcement learning with GRPO, sampling 8 completions per prompt, computing group-normalized advantages, and applying PPO-style clipped surrogate loss with KL penalty to a reference model (post-bootstrap).


Reward Function: Composite reward balancing:
Structure (0.2 if valid [bot]–[eot] span, -1.0 otherwise).
Correctness (1.0 for exact match, -0.5 otherwise; numerical tolerance for GSM8K).
Efficiency (-0.01 per reasoning token).
Gate (0.5 * mean inner gate - 0.2 * mean outer gate).
Clipped to [-1.5, 1.5] per component.


Datasets: GSM8K (math), ProsQA (professional QA), ProntoQA (logical QA), with tailored preprocessing (synthetic injection for Stage 1, raw questions for Stage 2).
Temperature Annealing: Exponential decay from 2.0 to 0.1, with halving on validation plateaus.

Evaluation

Metrics:
Accuracy: Percentage of correct answers.
Average Reasoning Steps: Number of tokens in reasoning span.
Gate Statistics: Mean inner/outer gates.


Validation: During training, checks structure rate and mean inner gate (Stage 1), average reward (Stage 2) on 256 held-out samples.

Implementation Details

Files:
model.py: Defines the custom DTT GPT-2 model with gating and blended embeddings.
reward.py: Computes the composite reward.
train.py: Entry point for two-stage training with Accelerate.
bootstrap.py: Stage 1 supervised bootstrap loop.
grpo.py: Stage 2 GRPO reinforcement loop.
datasets.py: Dataset loading and preprocessing (HF for GSM8K/ProntoQA, JSON for ProsQA).
utils.py: Validation helpers for structure/gate metrics and rewards.
YAML configs: Specify hyperparameters for each dataset (e.g., gsm8k.yaml).


Distributed Training: Uses Accelerate for DDP on 4 GPUs.
Monitoring: Console logging via tqdm; extendable to wandb if needed.

Running the Project

Setup:
Install: pip install -r requirements.txt.
Configure Accelerate: accelerate config (select multi-GPU, DDP).
Download ProsQA: Copy prosqa_*.json from Coconut GitHub to data/.


Training Commands:
GSM8K Stage 1: accelerate launch --num_processes 4 train.py --stage 1 --dataset gsm8k --config configs/train_gsm8k.yaml
GSM8K Stage 2: accelerate launch --num_processes 4 train.py --stage 2 --dataset gsm8k --config configs/gsm8k.yaml --ref_checkpoint bootstrap_checkpoint
ProsQA: Replace --dataset prosqa --config configs/prosqa.yaml.
ProntoQA: Replace --dataset prontoqa --config configs/prontoqa.yaml.



Conclusion
The DTT project successfully implements a language model with a dynamic gating mechanism for latent reasoning, drawing on COCONUT’s continuous thought approach. It supports distributed training and robust evaluation across multiple datasets without supervised CoT data. Future work could explore advanced annealing strategies or integration with larger models for further optimization.
Key Citations

Training Large Language Models to Reason in a Continuous Latent Space
