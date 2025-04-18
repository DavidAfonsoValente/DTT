import torch
from torch import nn
import torch.distributed as dist

class DTTModel(nn.Module):
    def __init__(self, base_causallm, bot_token_id, eot_token_id, eos_token_id, tokenizer):
        super().__init__()
        self.base_causallm = base_causallm
        self.embedding = base_causallm.get_input_embeddings()
        self.bot_token_id = bot_token_id
        self.eot_token_id = eot_token_id
        self.eos_token_id = eos_token_id
        self.tokenizer = tokenizer
        self.last_logits = []
        self.last_hidden_states = []
        self.epsilon_explore = 0.1  # Probability of adding noise
        self.noise_scale = 0.1     # Scale of Gaussian noise

    def forward(self, *args, **kwargs):
        return self.base_causallm(*args, **kwargs)

    def generate(
        self,
        input_ids,
        attention_mask,
        max_new_tokens,
        max_latent_steps,
        temperature,
        generations_per_prompt=4,  # Adjusted for 4 completions per GPU
    ):
        batch_size = input_ids.size(0)
        all_sequences = []
        all_latent_steps = []

        # Process each input in the batch
        for i in range(batch_size):
            sub_batch_input_ids = input_ids[i:i+1]  # [1, seq_len]
            sub_batch_attention_mask = attention_mask[i:i+1]  # [1, seq_len]
            sub_batch_output = self._generate_sub_batch(
                sub_batch_input_ids,
                sub_batch_attention_mask,
                max_new_tokens,
                max_latent_steps,
                temperature,
                generations_per_prompt,
            )
            all_sequences.extend(sub_batch_output['sequences'])
            all_latent_steps.extend(sub_batch_output['latent_steps'])

        # Stack sequences into a tensor
        max_len = max(len(seq) for seq in all_sequences)
        padded_sequences = []
        for seq in all_sequences:
            padding = [self.eos_token_id] * (max_len - len(seq))
            padded_seq = torch.cat((seq, torch.tensor(padding, device=input_ids.device)))
            padded_sequences.append(padded_seq)
        sequences_tensor = torch.stack(padded_sequences)

        return {
            'sequences': sequences_tensor,  # [batch_size * generations_per_prompt, max_len]
            'latent_steps': all_latent_steps  # List of ints
        }

    def _generate_sub_batch(self, input_ids, attention_mask, max_new_tokens, max_latent_steps, temperature, generations_per_input):
        device = input_ids.device
        sequences = []
        latent_steps_list = []

        for _ in range(generations_per_input):
            sequence = input_ids.clone().squeeze(0)  # [seq_len]
            mode = "token"
            latent_counter = 0
            finished = False
            self.last_hidden_states.append([])  # Per generation
            self.last_logits.append([])  # Per generation
            current_attention_mask = attention_mask.clone().squeeze(0)  # [seq_len]

            outputs = self.base_causallm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            last_hidden_state = outputs.hidden_states[-1][:, -1, :]
            past_key_values = outputs.past_key_values

            for step in range(max_new_tokens):
                if finished:
                    break

                if mode == "token":
                    last_token = sequence[-1].unsqueeze(0).unsqueeze(0)  # [1,1]
                    embed = self.embedding(last_token)
                else:  # latent mode
                    embed = last_hidden_state.unsqueeze(0)  # [1, embedding_dim]
                    if torch.rand(1, device=device).item() < self.epsilon_explore:
                        noise = torch.randn_like(embed) * self.noise_scale
                        embed = embed + noise

                input_embeds = embed.unsqueeze(0)  # [1,1,embedding_dim]

                current_attention_mask = torch.cat(
                    (current_attention_mask, torch.ones(1, device=device)),
                    dim=0
                )

                outputs = self.base_causallm(
                    inputs_embeds=input_embeds,
                    attention_mask=current_attention_mask.unsqueeze(0),
                    past_key_values=past_key_values,
                    output_hidden_states=True,
                )
                logits = outputs.logits[:, -1, :]  # [1, vocab_size]
                last_hidden_state = outputs.hidden_states[-1][:, -1, :]
                past_key_values = outputs.past_key_values

                if step == 0:
                    next_token = torch.tensor([self.bot_token_id], device=device)
                else:
                    if mode == "token":
                        logits = logits / temperature
                        probs = torch.softmax(logits, dim=-1)
                        next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
                    else:
                        virtual_token = torch.argmax(logits).item()
                        if virtual_token == self.eot_token_id or latent_counter >= max_latent_steps:
                            next_token = torch.tensor([self.eot_token_id], device=device)
                            mode = "token"
                        else:
                            latent_counter += 1
                            self.last_hidden_states[-1].append(last_hidden_state.clone())
                            self.last_logits[-1].append(logits.clone())
                            continue

                sequence = torch.cat((sequence, next_token), dim=0)
                if mode == "token":
                    if next_token.item() == self.bot_token_id:
                        mode = "latent"
                        latent_counter = 0
                    elif next_token.item() == self.eos_token_id:
                        finished = True

            sequences.append(sequence)
            latent_steps_list.append(latent_counter)

        return {
            'sequences': sequences,
            'latent_steps': latent_steps_list
        }