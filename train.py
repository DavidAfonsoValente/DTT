import yaml
from accelerate import Accelerator
from transformers import GPT2Config, GPT2Tokenizer
from src.model import DTTModel
from src.datasets import DTTDataset, collate_fn
from src.bootstrap import train_bootstrap
from src.grpo import train_grpo
import argparse
import torch
import wandb

parser = argparse.ArgumentParser()
parser.add_argument('--stage', type=int, choices=[1,2], required=True)
parser.add_argument('--dataset', type=str, choices=['gsm8k', 'prontoqa', 'prosqa'], required=True)
parser.add_argument('--config', type=str, required=True)
parser.add_argument('--ref_checkpoint', type=str, default=None)
parser.add_argument('--debug', action='store_true', default=False)
args = parser.parse_args()

accelerator = Accelerator()

with open(args.config, 'r') as f:
    config = yaml.safe_load(f)
config['dataset'] = args.dataset

# Init wandb on main process
if accelerator.is_local_main_process:
    wandb.init(project="dtt-training", config=config, name=f"{args.dataset}-stage{args.stage}")

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.add_special_tokens({
    'additional_special_tokens': ['[bot]', '[eot]'],
    'pad_token': '<pad>'
})
synthetic_ratio = 0.15 if args.stage == 1 else 0.0
dataset = DTTDataset(args.dataset, tokenizer, synthetic_ratio=synthetic_ratio, data_dir=config.get('data_dir', 'data'))
model = DTTModel.from_pretrained('gpt2', ignore_mismatched_sizes=True, low_cpu_mem_usage=True)

collate = lambda batch: collate_fn(batch, tokenizer.pad_token_id)

if args.stage == 1:
    train_bootstrap(model, dataset, config, accelerator, collate, tokenizer, debug=args.debug)
elif args.stage == 2:
    if args.ref_checkpoint is None:
        raise ValueError("Provide --ref_checkpoint for Stage 2")
    train_grpo(model, dataset, config, accelerator, args.ref_checkpoint, collate, debug=args.debug)

if accelerator.is_local_main_process:
    wandb.finish()