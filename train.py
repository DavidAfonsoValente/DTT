import yaml
from accelerate import Accelerator
from src.model import DTTModel
from src.datasets import DTTDataset
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
args = parser.parse_args()

accelerator = Accelerator()

with open(args.config, 'r') as f:
    config = yaml.safe_load(f)
config['dataset'] = args.dataset

# Init wandb on main process
if accelerator.is_local_main_process:
    wandb.init(project="dtt-training", config=config, name=f"{args.dataset}-stage{args.stage}")

tokenizer = DTTModel(GPT2Config.from_pretrained('gpt2')).tokenizer
dataset = DTTDataset(args.dataset, tokenizer, data_dir=config.get('data_dir', 'data'))
model = DTTModel.from_pretrained('gpt2')

if args.stage == 1:
    train_bootstrap(model, dataset, config, accelerator)
elif args.stage == 2:
    if args.ref_checkpoint is None:
        raise ValueError("Provide --ref_checkpoint for Stage 2")
    train_grpo(model, dataset, config, accelerator, args.ref_checkpoint)

if accelerator.is_local_main_process:
    wandb.finish()