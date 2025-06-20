import torch
import torch.nn as nn
from unsloth import FastLanguageModel

def gumbel_sigmoid(logits, temperature, hard=False, eps=1e-10):
    """Applies Gumbel-sigmoid to logits for near-binary gate values."""
    gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits, dtype=logits.dtype) + eps) + eps)
    y = (logits + gumbel_noise) / temperature
    y_soft = torch.sigmoid(y)
    
    if hard:
        y_hard = (y_soft > 0.5).float()
        y = (y_hard - y_soft).detach() + y_soft
    else:
        y = y_soft
    return y

class SparseGatedModel(FastLanguageModel):
    def __init__(self, model_name, max_seq_length, hidden_size, embedding_dim, lora_rank, gate_temperature, **kwargs):
        """Initializes the SparseGatedModel, extending Unsloth's FastLanguageModel for sparse gating."""
        super().__init__()
        
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            dtype=kwargs.get("dtype", None),
            load_in_4bit=kwargs.get("load_in_4bit", True),
        )
        
        self._hidden_size_internal = hidden_size
        self._embedding_dim_internal = embedding_dim
        
        self.projection = nn.Linear(self._hidden_size_internal, self._embedding_dim_internal, bias=False)
        self.gate_linear = nn.Linear(self._hidden_size_internal, 1)
        nn.init.constant_(self.gate_linear.bias, -3.0) 
        
        self.lora_rank_config = lora_rank
        self.gate_temperature = gate_temperature

        self.current_gate_values_for_batch = None
        self.current_original_embeddings_for_batch = None

        gpt2_target_modules = ["c_attn", "c_proj", "c_fc"]
        self.get_peft_model(
            r=self.lora_rank_config,
            target_modules=kwargs.get("lora_target_modules", gpt2_target_modules),
            lora_alpha=self.lora_rank_config * 2,
            lora_dropout=0.05,
            bias="none",
            use_gradient_checkpointing=kwargs.get("use_gradient_checkpointing", "unsloth"),
            random_state=3407,
            max_seq_length=max_seq_length,
            modules_to_save=["projection", "gate_linear"],
        )

    def forward(self,
                input_ids=None,
                attention_mask=None,
                inputs_embeds=None,
                past_key_values=None,
                prev_step_last_hidden_state=None,
                use_cache=True,
                output_attentions=False,
                output_hidden_states=True,
                gumbel_hard_during_forward=False,
                **kwargs):
        """
        Forward pass with HRPO-style gating.
        E'_t = Gate_t * Proj(H_{t-1}) + (1-Gate_t) * Emb(token_t)
        """
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("Cannot specify both input_ids and inputs_embeds.")
        
        if inputs_embeds is None:
            current_token_original_embeddings = self.model.get_input_embeddings()(input_ids)
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
            dummy_gate_logits = torch.full(
                (batch_size, seq_len, 1), -3.0, device=device, dtype=dtype
            )
            self.current_gate_values_for_batch = gumbel_sigmoid(
                dummy_gate_logits, self.gate_temperature, hard=gumbel_hard_during_forward
            ).detach().clone()

        outputs = self.model.forward(
            inputs_embeds=effective_embeddings,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            **kwargs
        )
        
        if not output_hidden_states or outputs.hidden_states is None:
            raise ValueError("output_hidden_states must be True.")
        
        last_hidden_state_current_step = outputs.hidden_states[-1]

        return outputs, last_hidden_state_current_step