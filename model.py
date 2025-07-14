import torch
import torch.nn as nn
from transformers import AutoTokenizer, PreTrainedModel
from peft import PeftModel, PeftConfig

def gumbel_sigmoid(logits: torch.Tensor, temperature: float, hard: bool = False, eps: float = 1e-10) -> torch.Tensor:
    """
    Applies the Gumbel-Sigmoid trick to get a differentiable approximation of a Bernoulli distribution.

    Args:
        logits: The raw logits from a model.
        temperature: The temperature for the Gumbel-Sigmoid distribution.
        hard: If True, returns a one-hot vector in the forward pass, while the backward pass uses the soft approximation.
        eps: A small epsilon value to prevent numerical instability.

    Returns:
        A tensor of near-binary gate values.
    """
    gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits, dtype=logits.dtype) + eps) + eps)
    y_soft = torch.sigmoid((logits + gumbel_noise) / temperature)

    if hard:
        y_hard = (y_soft > 0.5).float()
        # Straight-through estimator
        y = (y_hard - y_soft).detach() + y_soft
    else:
        y = y_soft
    return y

class SparseGatedModel(PeftModel):
    """
    A sparse gated model that dynamically decides whether to use a projected hidden state
    or the original token embedding for each token generation step.

    This class inherits from `peft.PeftModel` to be compatible with quantization
    and the `transformers.Trainer`.
    """
    def __init__(self, model: PreTrainedModel, peft_config: PeftConfig, hidden_size: int, embedding_dim: int, gate_temperature: float, **kwargs):
        """
        Initializes the SparseGatedModel.

        Args:
            model: The base transformer model (already quantized).
            peft_config: The configuration for PEFT (e.g., LoraConfig).
            hidden_size: The hidden size of the base model.
            embedding_dim: The embedding dimension of the base model.
            gate_temperature: The initial temperature for the Gumbel-Sigmoid gate.
        """
        # Initialize the PeftModel first, which handles adapter setup
        super().__init__(model, peft_config)

        self.tokenizer = AutoTokenizer.from_pretrained(model.config._name_or_path)

        # Custom trainable layers for the gating mechanism
        self.projection = nn.Linear(hidden_size, embedding_dim, bias=False)
        self.gate_linear = nn.Linear(hidden_size, 1)
        # Initialize gate bias to a negative value to encourage sparsity at the beginning of training
        nn.init.constant_(self.gate_linear.bias, -3.0)

        # Ensure the new layers are on the same device as the base model
        self.projection.to(self.device)
        self.gate_linear.to(self.device)

        # Model attributes
        self.gate_temperature = gate_temperature
        self.current_gate_values_for_batch = None
        self.current_original_embeddings_for_batch = None

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        past_key_values=None,
        prev_step_last_hidden_state=None,
        use_cache=True,
        output_attentions=False,
        output_hidden_states=True, # This must be True for the gating mechanism to work
        gumbel_hard_during_forward=False,
        return_last_hidden_state=False, # Flag to control the return signature
        logits_to_keep=None, # <--- ADD THIS ARGUMENT
        **kwargs
    ):
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")

        if inputs_embeds is None:
            current_token_original_embeddings = self.get_input_embeddings()(input_ids)
        else:
            current_token_original_embeddings = inputs_embeds

        self.current_original_embeddings_for_batch = current_token_original_embeddings.detach().clone()

        batch_size, seq_len, _ = current_token_original_embeddings.shape
        device = current_token_original_embeddings.device
        dtype = current_token_original_embeddings.dtype

        # The gating logic is applied only when the hidden state from the previous step is available
        if prev_step_last_hidden_state is not None:
            if prev_step_last_hidden_state.ndim == 2:
                prev_step_last_hidden_state = prev_step_last_hidden_state.unsqueeze(1)

            if prev_step_last_hidden_state.shape[1] == 1 and seq_len > 1:
                prev_step_last_hidden_state = prev_step_last_hidden_state.repeat(1, seq_len, 1)
            elif prev_step_last_hidden_state.shape[1] != seq_len:
                raise ValueError(
                    f"Shape mismatch: prev_step_last_hidden_state.shape[1] ({prev_step_last_hidden_state.shape[1]}) "
                    f"!= seq_len ({seq_len})"
                )

            gate_logits = self.gate_linear(prev_step_last_hidden_state)
            gate_values = gumbel_sigmoid(gate_logits, self.gate_temperature, hard=gumbel_hard_during_forward)
            projected_hidden = self.projection(prev_step_last_hidden_state)

            # Combine original embedding and projected hidden state based on the gate value
            effective_embeddings = gate_values * projected_hidden + (1 - gate_values) * current_token_original_embeddings
            self.current_gate_values_for_batch = gate_values.detach().clone()
        else:
            # For the initial prompt, use the original embeddings directly
            effective_embeddings = current_token_original_embeddings
            # Create dummy gate values for logging purposes
            dummy_gate_logits = torch.full((batch_size, seq_len, 1), -3.0, device=device, dtype=dtype)
            self.current_gate_values_for_batch = gumbel_sigmoid(
                dummy_gate_logits, self.gate_temperature, hard=gumbel_hard_during_forward
            ).detach().clone()

        # Call the forward method of the base model (which includes the PEFT adapters)
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
            raise ValueError("The base model must return hidden states (output_hidden_states=True).")

        last_hidden_state_current_step = outputs.hidden_states[-1]

        # Conditionally return the tuple required by the custom generation logic
        if return_last_hidden_state:
            return outputs, last_hidden_state_current_step
        else:
            return outputs