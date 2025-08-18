from accelerate import Accelerator
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
from src.utils import validate_bootstrap

def train_bootstrap(model, dataset, config, accelerator):
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, collate_fn=collate_fn)
    optimizer = AdamW(model.parameters(), lr=config['lr'])
    
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    
    for epoch in range(config['epochs']):
        model.train()
        for batch in tqdm(dataloader):
            outputs = model(input_ids=batch['input_ids'], labels=batch['input_ids'], noisy_mask=batch['noisy_mask'])
            loss = outputs['loss']
            
            gates = outputs['gates']
            noisy_mask = batch['noisy_mask']
            if noisy_mask is not None:
                inner_gates = gates[noisy_mask]
                mean_inner_gate = inner_gates.mean() if inner_gates.numel() > 0 else torch.tensor(0.0)
                gate_reg = config['lambda_gate'] * relu(0.7 - mean_inner_gate)
                loss += gate_reg
            
            accelerator.backward(loss)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
        
        if validate_bootstrap(model, config, accelerator, dataset.tokenizer):
            accelerator.save_state("bootstrap_checkpoint")
            break