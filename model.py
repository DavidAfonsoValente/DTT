import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig

def gumbel_sigmoid(logits: torch.Tensor, temperature: float, hard: bool = False, eps: float = 1e-10) -> torch.Tensor:
    """
    Applies Gumbel-Sigmoid for differentiable Bernoulli sampling.
    
    Args:
        logits: Raw logits from the model.
        temperature: Temperature for Gumbel-Sigmoid.
        hard: If True, uses hard decisions in forward pass with soft gradients.
        eps: Small value to prevent numerical instability.
    
    Returns:
        Tensor of near-binary gate values.
    """
    gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits, dtype=logits.dtype) + eps) + eps)
    y_soft = torch.sigmoid((logits + gumbel_noise) / temperature)
    if hard:
        y_hard = (y_soft > 0.5).float()
        y = (y_hard - y_soft).detach() + y_soft
    else:
        y = y_soft
    return y

class SparseGatedModel(PeftModel):
    """
    A model with a sparse gating mechanism to choose between projected hidden states
    and original embeddings for each token generation step.
    """
    def __init__(self, model: AutoModelForCausalLM, peft_config: PeftConfig, hidden_size: int, embedding_dim: int, gate_temperature: float):
        """
        Initializes the SparseGatedModel.
        
        Args:
            model: Base transformer model (quantized if applicable).
            peft_config: PEFT configuration (e.g., LoraConfig).
            hidden_size: Hidden state dimension of the base model.
            embedding_dim: Embedding dimension of the base model.
            gate_temperature: Initial temperature for Gumbel-Sigmoid gate.
        """
        super().__init__(model, peft_config)
        self.tokenizer = AutoTokenizer.from_pretrained(model.config._name_or_path)
        self.projection = nn.Linear(hidden_size, embedding_dim, bias=False)
        self.gate_linear = nn.Linear(hidden_size, 1)
        nn.init.constant_(self.gate_linear.bias, -3.0)
        self.gate_temperature = gate_temperature
        self.current_gate_values_for_batch = None

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        past_key_values=None,
        prev_step_last_hidden_state=None,
        use_cache=True,
        output_attentions=False,
        output_hidden_states=True,
        gumbel_hard_during_forward=False,
        return_last_hidden_state=False,
        **kwargs
    ):
        """
        Forward pass with gating mechanism.
        
        Args:
            input_ids: Token IDs.
            attention_mask: Attention mask.
            inputs_embeds: Precomputed embeddings (optional).
            past_key_values: Cached key/value pairs for generation.
            prev_step_last_hidden_state: Hidden state from previous step.
            use_cache: Whether to use cached key/values.
            output_attentions: Whether to return attention weights.
            output_hidden_states: Must be True for gating to work.
            gumbel_hard_during_forward: Use hard Gumbel-Sigmoid if True.
            return_last_hidden_state: Return hidden state with outputs if True.
        
        Returns:
            Model outputs, optionally with last hidden state.
        """
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("Cannot specify both input_ids and inputs_embeds")
        
        if inputs_embeds is None:
            current_token_original_embeddings = self.get_input_embeddings()(input_ids)
        else:
            current_token_original_embeddings = inputs_embeds
        
        self.current_original_embeddings_for_batch = current_token_original_embeddings.detach().clone()
        batch_size, seq_len, _ = current_token_original_embeddings.shape
        device = current_token_original_embeddings.device
        dtype = current_token_original_embeddings.dtype
        
        if prev_step_last_hidden_state is not None:
            if prev_step_last_hidden_state.ndim == 2:
                prev_step_last_hidden_state = prev_step_last_hidden_state.unsqueeze(1)
            if prev_step_last_hidden_state.shape[1] == 1 and seq_len > 1:
                prev_step_last_hidden_state = prev_step_last_hidden_state.repeat(1, seq_len, 1)
            elif prev_step_last_hidden_state.shape[1] != seq_len:
                raise ValueError(f"Shape mismatch: prev_step_last_hidden_state.shape[1] ({prev_step_last_hidden_state.shape[1]}) != seq_len ({seq_len})")
            
            gate_logits = self.gate_linear(prev_step_last_hidden_state)
            gate_values = gumbel_sigmoid(gate_logits, self.gate_temperature, hard=gumbel_hard_during_forward)
            projected_hidden = self.projection(prev_step_last_hidden_state)
            effective_embeddings = gate_values * projected_hidden + (1 - gate_values) * current_token_original_embeddings
            self.current_gate_values_for_batch = gate_values.detach().clone()
        else:
            effective_embeddings = current_token_original_embeddings
            dummy_gate_logits = torch.full((batch_size, seq_len, 1), -3.0, device=device, dtype=dtype)
            self.current_gate_values_for_batch = gumbel_sigmoid(dummy_gate_logits, self.gate_temperature, hard=gumbel_hard_during_forward).detach().clone()
        
        outputs = self.base_model(
            inputs_embeds=effective_embeddings,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            **kwargs
        )
        
        if not output_hidden_states or outputs.hidden_states is None:
            raise ValueError("Base model must return hidden states (output_hidden_states=True).")
        
        last_hidden_state_current_step = outputs.hidden_states[-1]
        if return_last_hidden_state:
            return outputs, last_hidden_state_current_step
        return outputs