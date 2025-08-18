import yaml
from accelerate import Accelerator
from src.model import DTTModel
from src.datasets import DTTDataset
from src.bootstrap import train_bootstrap
from src.grpo import train_grpo
import argparse
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--stage', type=int, choices=[1,2], required=True)
parser.add_argument('--dataset', type=str, choices=['gsm8k', 'prontoqa', 'prosqa'], required=True)
parser.add_argument('--config', type=str, required=True)
parser.add_argument('--ref_checkpoint', type=str, default=None)  # For Stage 2
args = parser.parse_args()

accelerator = Accelerator()

with open(args.config, 'r') as f:
    config = yaml.safe_load(f)
config['dataset'] = args.dataset  # Override

tokenizer = DTTModel(GPT2Config.from_pretrained('gpt2')).tokenizer
dataset = DTTDataset(args.dataset, tokenizer)
model = DTTModel.from_pretrained('gpt2')

if args.stage == 1:
    train_bootstrap(model, dataset, config, accelerator)
elif args.stage == 2:
    if args.ref_checkpoint is None:
        raise ValueError("Provide --ref_checkpoint for Stage 2")
    ref_model = DTTModel.from_pretrained(args.ref_checkpoint)
    train_grpo(model, dataset, config, accelerator)