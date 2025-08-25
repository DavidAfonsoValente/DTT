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


Forward Pass: Sequential with blending of token embeddings and hidden states based on gate values, using past_key_values for efficiency.
Generate Method: Custom generation loop that dynamically builds input embeddings for blended latent reasoning.

Training Process

Two-Stage Reward Curriculum in GRPO (No Supervised Bootstrap):
Stage 1 (Structure Foundation): Focuses on structure, gating, and basic correctness with simple rewards.
Stage 2 (Performance Optimization): Adds efficiency and stronger correctness, with automatic transition based on metrics.


Reward Function: Composite reward balancing:
Structure (varied by stage).
Correctness (basic in stage 1, exact in stage 2).
Efficiency (stage 2 only).
Gate (varied by stage).
Clipped to [-1.5, 1.5] per component.


Datasets: GSM8K (math), ProsQA (professional QA), ProntoQA (logical QA), with raw questions for GRPO.
Temperature Annealing: Per stage, exponential decay.

Evaluation

Metrics:
Accuracy: Percentage of correct answers.
Average Reasoning Steps: Number of tokens in reasoning span.
Gate Statistics: Mean inner/outer gates.


Validation: During training, checks structure rate, gate ratio, basic accuracy for transition.

Implementation Details

Files:
model.py: Defines the custom DTT GPT-2 model with gating and blended embeddings.
rewards.py: Computes the stage-specific composite reward.
train.py: Entry point for GRPO training with curriculum.
grpo.py: GRPO reinforcement loop with stages.
datasets.py: Dataset loading and preprocessing (HF for GSM8K/ProntoQA, JSON for ProsQA).
utils.py: Validation helpers for transition criteria and metrics.
YAML configs: Specify hyperparameters for each dataset (e.g., gsm8k.yaml).


Distributed Training: Uses Accelerate for DDP on 4 GPUs.
Monitoring: Console logging via tqdm; WandB integrated. Checkpoints saved every 1000 steps.

Running the Project

Setup:
Install: pip install -r requirements.txt.
Configure Accelerate: accelerate config (follow prompts for multi-GPU).
Download ProsQA: Copy prosqa_*.json from Coconut GitHub to data/.


Training Commands:
GSM8K: accelerate launch --num_processes 4 train.py --dataset gsm8k --config configs/gsm8k.yaml
ProsQA: accelerate launch --num_processes 4 train.py --dataset prosqa --config configs/prosqa.yaml
ProntoQA: accelerate launch --num_processes 4 train.py --dataset prontoqa --config configs/prontoqa.yaml


Conclusion
The DTT project successfully implements a language model with a dynamic gating mechanism for latent reasoning, drawing on COCONUT’s continuous thought approach. It supports distributed training and robust evaluation across multiple datasets without supervised CoT data. Future work could explore advanced annealing strategies or integration with larger models for further optimization.
Key Citations

Training Large Language Models to Reason in a Continuous Latent Space