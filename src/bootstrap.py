from accelerate import Accelerator
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
from src.utils import validate_bootstrap
import wandb
from torch.nn.functional import relu
import time
import math

def train_bootstrap(model, dataset, config, accelerator, collate_fn, tokenizer, debug=False):
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, collate_fn=collate_fn)
    optimizer = AdamW(model.parameters(), lr=config['lr'])
    accum_steps = config.get('accum_steps', 1)
    
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    
    if debug:
        accelerator.unwrap_model(model).debug = True
    
    if accelerator.is_local_main_process:
        wandb.log({"stage": "bootstrap_start", "epoch": 0})
    
    global_step = 0
    
    for epoch in range(config['epochs']):
        model.train()
        epoch_loss = 0.0
        epoch_gate_reg = 0.0
        num_batches = 0
        optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(tqdm(dataloader)):
            if debug and accelerator.is_local_main_process:
                input_text = tokenizer.decode(batch['input_ids'][0], skip_special_tokens=False)
                print(f"Decoded input text: {input_text[:512] if len(input_text) > 512 else input_text}")
                if batch['labels'] is not None:
                    labels_text = tokenizer.decode(batch['labels'][0][batch['labels'][0] != -100], skip_special_tokens=False)
                    print(f"Decoded labels text: {labels_text[:512] if len(labels_text) > 512 else labels_text}")
            
            outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['labels'], noisy_mask=batch['noisy_mask'])
            loss = outputs.loss
            
            gates = outputs.gates
            noisy_mask = batch['noisy_mask']
            gate_reg_value = 0.0
            if noisy_mask is not None:
                inner_gates = gates[noisy_mask]
                outer_gates = gates[~noisy_mask]
                mean_inner_gate = inner_gates.mean() if inner_gates.numel() > 0 else torch.tensor(0.0)
                mean_outer_gate = outer_gates.mean() if outer_gates.numel() > 0 else torch.tensor(0.0)
                gate_reg_inner = config['lambda_gate'] * relu(0.7 - mean_inner_gate)
                gate_reg = gate_reg_inner
                loss += gate_reg
                gate_reg_value = gate_reg.item()
                if debug and accelerator.is_local_main_process:
                    print(f"Mean inner gate: {mean_inner_gate.item():.4f}")
                    print(f"Mean outer gate: {mean_outer_gate.item():.4f}")
                    print(f"Gate regularization: {gate_reg_value:.4f}")
            
            loss = loss / accum_steps
            accelerator.backward(loss)
            
            if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(dataloader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
                if global_step % 100 == 0:
                    new_tau = max(0.1, 2.0 * (0.9 ** (global_step / 1000)))
                    model.set_temperature(new_tau)
            
            epoch_loss += loss.item() * accum_steps
            epoch_gate_reg += gate_reg_value
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches
        avg_gate_reg = epoch_gate_reg / num_batches
        
        if accelerator.is_local_main_process:
            wandb.log({
                "bootstrap/epoch": epoch + 1,
                "bootstrap/loss": avg_loss,
                "bootstrap/gate_reg": avg_gate_reg
            })
        
        if debug and accelerator.is_local_main_process:
            sample_batch = next(iter(dataloader))
            sample_input = sample_batch['input_ids'][:1]
            with torch.no_grad():
                sample_outputs = model(input_ids=sample_input, attention_mask=sample_input.ne(tokenizer.pad_token_id))
            gate_after_prompt = sample_outputs.gates[0, -1].item()
            logit_bot = sample_outputs.logits[0, -1, model.bot_id].item()
            prob_bot = torch.softmax(sample_outputs.logits[0, -1], dim=-1)[model.bot_id].item()
            print(f"Diagnostic on sample prompt: Gate after prompt: {gate_after_prompt:.4f}, Logit for [bot]: {logit_bot:.4f}, Prob for [bot]: {prob_bot:.4f}")
        
        valid = validate_bootstrap(model, config, accelerator, tokenizer, debug=debug)
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