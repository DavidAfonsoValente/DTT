import torch
import torch.nn as nn
from unsloth import FastLanguageModel

def gumbel_sigmoid(logits, temperature, hard=False):
    """Apply Gumbel-sigmoid to logits for near-binary gate values."""
    gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits)))
    y = (logits + gumbel_noise) / temperature
    y = torch.sigmoid(y)
    if hard:
        y_hard = (y > 0.5).float()
        y = y_hard - y.detach() + y
    return y

class SparseGatedModel(FastLanguageModel):
    def __init__(self, model_name="gpt2", hidden_size=768, embedding_dim=768, lora_rank=32, temperature=1.0):
        """Initialize the model with projection and gate layers."""
        super().__init__(model_name)
        self.hidden_size = hidden_size
        self.embedding_dim = embedding_dim
        self.projection = nn.Linear(hidden_size, embedding_dim)
        self.gate_linear = nn.Linear(hidden_size, 1)
        self.lora_rank = lora_rank
        self.temperature = temperature
        self.setup_lora()

    def setup_lora(self):
        """Configure LoRA for efficient fine-tuning."""
        self.get_peft_model(
            r=self.lora_rank,
            target_modules=["c_attn", "c_proj", "c_fc"],
            modules_to_save=["projection", "gate_linear"],
            lora_alpha=self.lora_rank * 2,
            use_gradient_checkpointing="unsloth"
        )

    def forward(self, input_ids, attention_mask=None, past_key_values=None, **kwargs):
        """Forward pass with sparse gating."""
        embeddings = self.get_input_embeddings()(input_ids)
        outputs = super().forward(
            inputs_embeds=embeddings,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            output_hidden_states=True,
            **kwargs
        )
        hidden_states = outputs.hidden_states[-1]
        projected_hidden = self.projection(hidden_states)
        gate_logits = self.gate_linear(hidden_states)
        gate_values = gumbel_sigmoid(gate_logits, self.temperature, hard=False)
        combined_input = gate_values * projected_hidden + (1 - gate_values) * embeddings
        outputs.combined_input = combined_input
        outputs.gate_values = gate_values
        return outputs