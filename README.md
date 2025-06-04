# Discrete Thinking Tokens (DTT) Project Description

## Introduction

The Discrete Thinking Tokens (DTT) project aims to train a GPT-2-based language model to generate structured outputs for reasoning tasks, producing sequences in the format `[bot]`...reasoning...`[eot]`...answer. It employs a sparse gating mechanism to decide when to use hidden states for latent reasoning (gate ≈ 1) or token embeddings (gate ≈ 0), inspired by the Hybrid Reasoning Policy Optimization (HRPO) framework ([arXiv:2505.18454](https://arxiv.org/abs/2505.18454)) and the Chain of Continuous Thought (COCONUT) method ([arXiv:2412.06769](https://arxiv.org/abs/2412.06769)). The project supports datasets like GSM8K, ProsQA, and ProntoQA, uses Group Relative Policy Optimization (GRPO) for training, and is optimized for distributed training on 4 GPUs with comprehensive monitoring via Weights & Biases (wandb).

## Background

### HRPO: Hybrid Reasoning Policy Optimization
HRPO, introduced in the paper *"Hybrid Latent Reasoning via Reinforcement Learning"* ([arXiv:2505.18454](https://arxiv.org/abs/2505.18454)), integrates prior hidden states into sampled tokens using a learnable gating mechanism. It initializes training with predominantly token embeddings and progressively incorporates hidden features, balancing the discrete nature of language generation with continuous reasoning representations. This hybrid approach enhances LLMs’ reasoning capabilities while maintaining generative performance, making it suitable for tasks requiring multi-step reasoning.

### COCONUT: Chain of Continuous Thought
COCONUT, described in *"Training Large Language Models to Reason in a Continuous Latent Space"* ([arXiv:2412.06769](https://arxiv.org/abs/2412.06769)), enables LLMs to reason in a continuous latent space by using the last hidden state as a reasoning state, fed back as the next input embedding. This avoids generating intermediate text tokens, improving efficiency for tasks like mathematical reasoning or logical puzzles that require backtracking. COCONUT’s approach is particularly effective for complex reasoning, as it reduces the overhead of textual coherence.

## Project Description

The DTT project combines HRPO’s hybrid reasoning and COCONUT’s focus on latent efficiency to create a model that generates structured outputs with minimal reasoning tokens. The model uses a Gumbel-sigmoid gating mechanism to control hidden state usage, trained with GRPO to optimize correctness, efficiency, and gate sparsity.

### Model Architecture
- **SparseGatedModel**: Extends Unsloth’s `FastLanguageModel` for GPT-2, with:
  - **Projection Layer**: Maps hidden states (768 dimensions) to the embedding space.
  - **Gate Linear Layer**: Produces gate logits from hidden states.
  - **Gumbel-Sigmoid**: Ensures near-binary gate values for sparsity, controlled by a temperature parameter.
  - **LoRA**: Fine-tunes attention and feed-forward layers efficiently.
- **Forward Pass**: Combines token embeddings and projected hidden states based on gate values, storing results for reward computation.

### Training Process
- **CustomGRPOTrainer**: A subclass of TRL’s `GRPOTrainer`, supporting:
  - Distributed training on 4 GPUs using Hugging Face’s Accelerate library.
  - Sequence generation with `model.generate`, collecting hidden states for gate value computation.
  - Detailed wandb logging of reward components, gate statistics, and a progress table.
- **Reward Function**: Balances:
  - Correctness (1.0 for correct answers, 0.0 otherwise).
  - Token efficiency (penalizes reasoning token count).
  - Gate sparsity (encourages gate ≈ 1 during reasoning, ≈ 0 elsewhere).
- **Datasets**: GSM8K (math), ProsQA (professional), ProntoQA (logical), with tailored preprocessing.

### Evaluation
- **Metrics**:
  - Accuracy: Percentage of correct answers.
  - Average Latent Steps: Number of reasoning tokens.
  - Gate Statistics: Mean, std, min, max of gate values inside/outside reasoning.
- **Visualizations**: Wandb logs include gate value histograms, reasoning length histograms, and a per-example metrics table.
- **Artifacts**: Hidden states and gate values saved as `.pt` files.

## Implementation Details
- **Files**:
  - `model.py`: Defines the model with Gumbel-sigmoid gating.
  - `reward.py`: Computes rewards for correctness, efficiency, and sparsity.
  - `train.py`: Manages training with distributed support and wandb logging.
  - `eval.py`: Evaluates performance with detailed metrics.
  - `utils.py`: Handles dataset preprocessing and answer comparison.
  - YAML configs: Specify hyperparameters for each dataset.
- **Distributed Training**: Launched with `accelerate launch --num_processes 4 train.py --config_path <config>`.
- **Monitoring**: Wandb tracks reward components, gate statistics, and progress tables.

## Running the Project
- **Setup**:
  - Install: `pip install accelerate transformers trl unsloth datasets torch wandb pyyaml`.
  - Configure Accelerate: `accelerate config` (select DDP, 4 GPUs).
- **Training Commands**:
  - GSM8K: `accelerate launch --num_processes 4 train.py --config_path configs/train_gsm8k.yaml`
  - ProsQA: `accelerate launch --num_processes 4 train.py --config_path configs/train_prosqa.yaml`
  - ProntoQA: `accelerate launch --num_processes 4 train.py --config_path configs/train_prontoqa.yaml`
- **Evaluation**: `python eval.py --config_path configs/eval_gsm8k.yaml` (adjust for other datasets).

## Conclusion
The DTT project appears to successfully implement a language model with a sparse gating mechanism, drawing on HRPO’s hybrid reasoning and COCONUT’s latent efficiency. It supports distributed training, comprehensive monitoring, and robust evaluation across multiple datasets. Future work could explore advanced gating techniques or continuous thought integration for further optimization.

## Key Citations
- [Hybrid Latent Reasoning via Reinforcement Learning](https://arxiv.org/abs/2505.18454)
- [Training Large Language Models to Reason in a Continuous Latent Space](https://arxiv.org/abs/2412.06769)