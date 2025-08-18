# src/datasets.py
from datasets import load_dataset
from torch.utils.data import Dataset
import random
import torch
import os
from torch.nn.utils.rnn import pad_sequence

class DTTDataset(Dataset):
    def __init__(self, dataset_name, tokenizer, split='train', synthetic_ratio=0.15, data_dir='data'):
        self.tokenizer = tokenizer
        self.synthetic_ratio = synthetic_ratio
        self.vocab_size = len(tokenizer) - 3  # Adjusted for added tokens
        self.bot_id = tokenizer.convert_tokens_to_ids('[bot]')
        self.eot_id = tokenizer.convert_tokens_to_ids('[eot]')
        self.data_dir = data_dir

        if dataset_name == 'gsm8k':
            self.data = load_dataset('gsm8k', 'main', split=split)
        elif dataset_name == 'prontoqa':
            self.data = load_dataset('allenai/prontoqa', split=split)
        elif dataset_name == 'prosqa':
            file_path = os.path.join(data_dir, f'prosqa_{split}.json')
            if not os.path.exists(file_path):
                raise ValueError(f"ProsQA file {file_path} not found. Download prosqa_{split}.json from https://github.com/facebookresearch/coconut/data")
            self.data = load_dataset('json', data_files=file_path, split='train')
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        question = item['question']['text'] if isinstance(item['question'], dict) else item['question']
        answer = item['answer'] if 'answer' in item else (item['answers'][0] if 'answers' in item else item.get('explanation', ''))

        input_ids = self.tokenizer.encode(question, return_tensors='pt').squeeze()

        if random.random() < self.synthetic_ratio:
            k = random.randint(0, 5)
            fillers_ids = [random.randint(0, self.vocab_size - 1) for _ in range(k)]
            fillers_text = ' '.join(self.tokenizer.decode([f]) for f in fillers_ids)
            input_text = f"{question} [bot] {fillers_text} [eot] {answer}"
            input_ids = self.tokenizer.encode(input_text, return_tensors='pt').squeeze()
            bot_pos = (input_ids == self.bot_id).nonzero(as_tuple=True)[0][0].item() if len((input_ids == self.bot_id).nonzero()) > 0 else -1
            eot_pos = (input_ids == self.eot_id).nonzero(as_tuple=True)[0][0].item() if len((input_ids == self.eot_id).nonzero()) > 0 else -1
            noisy_mask = torch.zeros_like(input_ids, dtype=torch.bool)
            if bot_pos != -1 and eot_pos != -1 and bot_pos < eot_pos:
                noisy_mask[bot_pos + 1:eot_pos] = True
            labels = input_ids.clone()
        else:
            noisy_mask = torch.zeros_like(input_ids, dtype=torch.bool)
            labels = input_ids.clone()

        return {'input_ids': input_ids, 'labels': labels, 'noisy_mask': noisy_mask, 'answer_gt': answer}

def collate_fn(batch, pad_token_id, ignore_index=-100):
    input_ids = pad_sequence([b['input_ids'] for b in batch], batch_first=True, padding_value=pad_token_id)
    labels = pad_sequence([b['labels'] for b in batch], batch_first=True, padding_value=ignore_index)
    noisy_mask = pad_sequence([b['noisy_mask'] for b in batch], batch_first=True, padding_value=False)
    attention_mask = input_ids.ne(pad_token_id)
    return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels, 'noisy_mask': noisy_mask, 'answer_gt': [b['answer_gt'] for b in batch]}