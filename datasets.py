from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
import random
import torch
import os
import json

class DTTDataset(Dataset):
    def __init__(self, dataset_name, tokenizer, split='train', synthetic_ratio=0.15, data_dir='data'):
        self.tokenizer = tokenizer
        self.synthetic_ratio = synthetic_ratio
        self.vocab_size = len(tokenizer) - 2
        self.bot_id = tokenizer.convert_tokens_to_ids('[bot]')
        self.eot_id = tokenizer.convert_tokens_to_ids('[eot]')
        
        if dataset_name == 'gsm8k':
            self.data = load_dataset('gsm8k', 'main', split=split)
        elif dataset_name == 'prontoqa':
            self.data = load_dataset('allenai/prontoqa', split=split)
        elif dataset_name == 'prosqa':
            # Custom from Coconut: Assume downloaded to data/prosqa_train.json etc.
            file_path = os.path.join(data_dir, f'prosqa_{split}.json')
            if not os.path.exists(file_path):
                raise ValueError(f"Download prosqa_{split}.json from https://github.com/facebookresearch/coconut/data")
            self.data = load_dataset('json', data_files=file_path, split='train')  # All as train split
        else:
            raise ValueError("Unknown dataset")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        question = item['question']['text'] if isinstance(item['question'], dict) else item['question']
        answer = item['answer'] if 'answer' in item else (item['answers'][0] if 'answers' in item else item.get('explanation', ''))

        if random.random() < self.synthetic_ratio:
            k = random.randint(0, 5)
            fillers_ids = [random.randint(0, self.vocab_size - 1) for _ in range(k)]
            fillers_text = ' '.join(self.tokenizer.decode([f]) for f in fillers_ids)
            input_text = f"{question} [bot] {fillers_text} [eot] {answer}"
            input_ids = self.tokenizer.encode(input_text, return_tensors='pt').squeeze()
            bot_pos = (input_ids == self.bot_id).nonzero(as_tuple=True)[0][0].item() if len((input_ids == self.bot_id).nonzero()) > 0 else -1
            eot_pos = (input_ids == self.eot_id).nonzero(as_tuple=True)[0][0].item() if len((input_ids == self.eot_id).nonzero()) > 0 else -1
            noisy_mask = torch.zeros_like(input_ids, dtype=torch.bool)
            if bot_pos != -1 and eot_pos != -1:
                noisy_mask[bot_pos + 1:eot_pos] = True
            return {'input_ids': input_ids, 'labels': input_ids.clone(), 'noisy_mask': noisy_mask, 'answer_gt': answer}

        input_ids = self.tokenizer.encode(question, return_tensors='pt').squeeze()
        return {'input_ids': input_ids, 'answer_gt': answer}

def collate_fn(batch):
    input_ids = [b['input_ids'] for b in batch]
    max_len = max(len(ids) for ids in input_ids)
    padded_ids = torch.stack([torch.cat([ids, torch.full((max_len - len(ids),), 0, dtype=torch.long)]) for ids in input_ids])
    noisy_mask = torch.stack([torch.cat([b.get('noisy_mask', torch.zeros_like(b['input_ids'])), torch.full((max_len - len(b['input_ids']),), False)]) for b in batch]) if 'noisy_mask' in batch[0] else None
    return {'input_ids': padded_ids, 'noisy_mask': noisy_mask, 'answer_gt': [b['answer_gt'] for b in batch]}