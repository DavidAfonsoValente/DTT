from accelerate import Accelerator
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
from src.utils import validate_bootstrap
import wandb
from torch.nn.functional import relu
import time

def train_bootstrap(model, dataset, config, accelerator, collate_fn, debug=False):
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, collate_fn=collate_fn)
    optimizer = AdamW(model.parameters(), lr=config['lr'])
    
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    
    if accelerator.is_local_main_process:
        wandb.log({"stage": "bootstrap_start", "epoch": 0})
    
    for epoch in range(config['epochs']):
        if debug and accelerator.is_local_main_process:
            print(f"Starting epoch {epoch + 1}/{config['epochs']}")
            epoch_start = time.time()
        
        model.train()
        epoch_loss = 0.0
        epoch_gate_reg = 0.0
        num_batches = 0
        
        for batch in tqdm(dataloader):
            if debug and accelerator.is_local_main_process:
                batch_start = time.time()
            
            outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['labels'], noisy_mask=batch['noisy_mask'])
            loss = outputs.loss
            
            gates = outputs.gates
            noisy_mask = batch['noisy_mask']
            gate_reg_value = 0.0
            if noisy_mask is not None:
                inner_gates = gates[noisy_mask]
                mean_inner_gate = inner_gates.mean() if inner_gates.numel() > 0 else torch.tensor(0.0)
                gate_reg = config['lambda_gate'] * relu(0.7 - mean_inner_gate)
                loss += gate_reg
                gate_reg_value = gate_reg.item()
            
            accelerator.backward(loss)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            
            epoch_loss += loss.item()
            epoch_gate_reg += gate_reg_value
            num_batches += 1
            
            if debug and accelerator.is_local_main_process:
                print(f"Batch processed in {time.time() - batch_start:.2f}s | Loss: {loss.item():.4f}")
        
        avg_loss = epoch_loss / num_batches
        avg_gate_reg = epoch_gate_reg / num_batches
        
        if accelerator.is_local_main_process:
            wandb.log({
                "bootstrap/epoch": epoch + 1,
                "bootstrap/loss": avg_loss,
                "bootstrap/gate_reg": avg_gate_reg
            })
        
        if debug and accelerator.is_local_main_process:
            print(f"Epoch {epoch + 1} completed in {time.time() - epoch_start:.2f}s | Avg Loss: {avg_loss:.4f}")
        
        valid = validate_bootstrap(model, config, accelerator, dataset.tokenizer, debug=debug)
        if accelerator.is_local_main_process:
            wandb.log({
                "bootstrap/validation_structure_rate": valid['structure_rate'],
                "bootstrap/validation_mean_inner_gate": valid['mean_inner_gate']
            })
        
        if valid['is_valid']:
            if accelerator.is_local_main_process:
                wandb.log({"bootstrap/completed": True})
            accelerator.save_state("bootstrap_checkpoint")
            break