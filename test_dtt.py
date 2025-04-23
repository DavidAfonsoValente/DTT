"""
Test script for Dynamic Thinking Tokens (DTT) Framework

This script provides a simplified test to verify the DTT implementation
without requiring distributed training.
"""

import torch
import os
import sys
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add paths for imports
sys.path.append('/home/ubuntu/dtt_package')

# Import DTT modules
from dtt import DTT

def main():
    print("Testing DTT implementation...")
    
    # Load model and tokenizer
    print("Loading model and tokenizer...")
    model_id = "openai-community/gpt2"
    model = AutoModelForCausalLM.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Add special tokens for latent reasoning
    print("Adding special tokens...")
    tokenizer.add_tokens("<|start-latent|>")
    tokenizer.add_tokens("<|end-latent|>")
    tokenizer.add_tokens("<|latent|>")
    latent_id = tokenizer.convert_tokens_to_ids("<|latent|>")
    start_id = tokenizer.convert_tokens_to_ids("<|start-latent|>")
    end_id = tokenizer.convert_tokens_to_ids("<|end-latent|>")
    
    # Resize token embeddings
    print("Resizing token embeddings...")
    model.resize_token_embeddings(len(tokenizer))
    
    # Initialize embeddings for special tokens
    embeddings = model.get_input_embeddings()
    target_id = tokenizer.convert_tokens_to_ids("<<")
    for token_id in [latent_id, start_id, end_id]:
        embeddings.weight.data[token_id] = embeddings.weight.data[target_id]
        lm_head = model.lm_head
        lm_head.weight.data[token_id] = lm_head.weight.data[target_id]
    
    # Wrap model with DTT
    print("Creating DTT model...")
    dtt_model = DTT(model, latent_id, start_id, end_id, tokenizer.eos_token_id)
    
    # Test different phases
    print("\nTesting phase settings...")
    for phase in ["warmup", "core", "final"]:
        dtt_model.set_phase(phase)
        print(f"Phase: {phase}")
        print(f"  Binary weight: {dtt_model.w_binary}")
        print(f"  CRS weight: {dtt_model.w_crs}")
        print(f"  LCR weight: {dtt_model.w_lcr}")
        print(f"  EDE weight: {dtt_model.w_ede}")
        print(f"  Max latent steps: {dtt_model.max_latent_steps}")
    
    # Test forward pass with a simple input
    print("\nTesting forward pass...")
    
    # Create a simple input with latent tokens
    text = "What is 2+3? <|start-latent|><|latent|><|latent|><|end-latent|> The answer is 5."
    inputs = tokenizer(text, return_tensors="pt")
    
    # Run forward pass
    outputs = dtt_model(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        labels=inputs["input_ids"],
        position_ids=torch.arange(0, inputs["input_ids"].shape[1]).unsqueeze(0),
        compute_rewards=True
    )
    
    # Check outputs
    print(f"Loss: {outputs.loss.item()}")
    print(f"Logits shape: {outputs.logits.shape}")
    if outputs.rewards is not None:
        print(f"Rewards: {outputs.rewards.item()}")
    if outputs.advantages is not None:
        print(f"Advantages: {outputs.advantages.item()}")
    
    # Test generation
    print("\nTesting generation...")
    prompt = "What is 2+3?"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    attention_mask = torch.ones_like(input_ids)
    
    generated_ids = dtt_model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=20
    )
    
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    print(f"Prompt: {prompt}")
    print(f"Generated: {generated_text}")
    
    print("\nDTT implementation test completed successfully!")

if __name__ == "__main__":
    main()
