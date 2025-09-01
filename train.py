# main.py
import yaml
from accelerate import Accelerator
from transformers import GPT2Tokenizer
from src.model import DTTModel
from src.datasets import DTTDataset, collate_fn
from src.grpo import train_grpo
import argparse
import torch
import wandb
import os
os.environ["TORCH_COMPILE_DISABLE"] = "1"  # Fully disables compile
os.environ["TORCHINDUCTOR_MAX_AUTOTUNE"] = "0"  # Fallback if partially fails

from datetime import timedelta
from accelerate.state import PartialState

#import torch._dynamo as dynamo
#dynamo.config.cache_size_limit = 64
#dynamo.config.suppress_errors = True

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, choices=['gsm8k', 'prontoqa', 'prosqa'], required=True)
parser.add_argument('--config', type=str, required=True)
parser.add_argument('--debug', action='store_true', default=False)
args = parser.parse_args()

accelerator = Accelerator(mixed_precision="fp16")  # Enable mixed precision for efficiency

if torch.cuda.is_available():
    print(f"Using CUDA device {torch.cuda.current_device()}", flush=True)

with open(args.config, 'r') as f:
    config = yaml.safe_load(f)
config['dataset'] = args.dataset

if accelerator.is_local_main_process:
    wandb.init(project="dtt-training", config=config, name=f"{args.dataset}-grpo")

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.add_special_tokens({
    'additional_special_tokens': ['[bot]', '[eot]'],
    'pad_token': '<pad>'
})
dataset = DTTDataset(args.dataset, tokenizer, data_dir=config.get('data_dir', 'data'))
model = DTTModel.from_pretrained('gpt2', ignore_mismatched_sizes=True)
ref_model = DTTModel.from_pretrained('gpt2', ignore_mismatched_sizes=True)
ref_model.load_state_dict(model.state_dict())

collate = lambda batch: collate_fn(batch, tokenizer.pad_token_id)

os.makedirs('checkpoints', exist_ok=True)
model, ref_model, dataset = accelerator.prepare(model, ref_model, dataset)
train_grpo(model, ref_model, dataset, config, accelerator, collate, tokenizer, debug=args.debug)

if accelerator.is_local_main_process:
    wandb.finish()