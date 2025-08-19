from accelerate import Accelerator
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
from src.utils import validate_bootstrap
import wandb
from torch.nn.functional import relu
import time

def train_bootstrap(model, dataset, config, accelerator, collate_fn, tokenizer, debug=False):
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, collate_fn=collate_fn)
    optimizer = AdamW(model.parameters(), lr=config['lr'])
    accum_steps = config.get('accum_steps', 1)  # Default to 1 if not specified
    
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    
    if debug:
        accelerator.unwrap_model(model).debug = True
    
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
        optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(tqdm(dataloader)):
            if debug and accelerator.is_local_main_process:
                batch_start = time.time()
                print(f"Processing batch {batch_idx + 1}/{len(dataloader)}")
                print(f"Batch input_ids shape: {batch['input_ids'].shape}")
                input_text = tokenizer.decode(batch['input_ids'][0], skip_special_tokens=False)
                print(f"Decoded input text: {input_text[:200]}...")
                if batch['labels'] is not None:
                    labels_text = tokenizer.decode(batch['labels'][0][batch['labels'][0] != -100], skip_special_tokens=False)
                    print(f"Decoded labels text: {labels_text[:200]}...")
                print(f"Batch noisy_mask shape: {batch['noisy_mask'].shape if batch['noisy_mask'] is not None else 'None'}")
            
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
                if debug and accelerator.is_local_main_process:
                    print(f"Mean inner gate: {mean_inner_gate.item():.4f}")
                    print(f"Gate regularization: {gate_reg_value:.4f}")
            
            # Scale loss for accumulation
            loss = loss / accum_steps
            accelerator.backward(loss)
            
            if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(dataloader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                if debug and accelerator.is_local_main_process:
                    print(f"Gradient update after {accum_steps} accumulations")
            
            epoch_loss += loss.item() * accum_steps  # Unscale for logging
            epoch_gate_reg += gate_reg_value
            num_batches += 1
            
            if debug and accelerator.is_local_main_process:
                print(f"Batch processed in {time.time() - batch_start:.2f}s | Final Loss: {loss.item() * accum_steps:.4f}")
        
        avg_loss = epoch_loss / num_batches
        avg_gate_reg = epoch_gate_reg / num_batches
        
        if accelerator.is_local_main_process:
            wandb.log({
                "bootstrap/epoch": epoch + 1,
                "bootstrap/loss": avg_loss,
                "bootstrap/gate_reg": avg_gate_reg
            })
        
        if debug and accelerator.is_local_main_process:
            print(f"Epoch {epoch + 1} completed in {time.time() - epoch_start:.2f}s | Avg Loss: {avg_loss:.4f} | Avg Gate Reg: {avg_gate_reg:.4f}")
        
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