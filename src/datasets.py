from torch.utils.data import Dataset
from datasets import load_dataset
import os
from torch.nn.utils.rnn import pad_sequence

class DTTDataset(Dataset):
    def __init__(self, dataset_name, tokenizer, split='train', data_dir='data'):
        self.tokenizer = tokenizer
        self.dataset_name = dataset_name
        self.data_dir = data_dir
        self.max_prompt_len = 128  # Fixed length; can move to config if desired
        if split == 'valid':
            split = 'test' if dataset_name in ['gsm8k', 'prontoqa'] else 'validation'

        if dataset_name == 'gsm8k':
            self.data = load_dataset('gsm8k', 'main', split=split)
        elif dataset_name == 'prontoqa':
            self.data = load_dataset('allenai/prontoqa', split=split)
        elif dataset_name == 'prosqa':
            file_path = os.path.join(data_dir, f'prosqa_{split}.json')
            if not os.path.exists(file_path):
                raise ValueError(f"ProsQA file {file_path} not found.")
            self.data = load_dataset('json', data_files=file_path, split='train')
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        question = item['question']['text'] if isinstance(item['question'], dict) else item['question']
        answer = item['answer'] if 'answer' in item else (item['answers'][0] if 'answers' in item else item.get('explanation', ''))
        answer_gt = answer.split('####')[-1].strip() if self.dataset_name == 'gsm8k' else answer

        enc = self.tokenizer(question, max_length=self.max_prompt_len, truncation=True, padding='max_length', return_tensors='pt')
        input_ids = enc['input_ids'].squeeze()
        attention_mask = enc['attention_mask'].squeeze()

        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'answer_gt': answer_gt}

def collate_fn(batch, pad_token_id):
    input_ids = torch.stack([b['input_ids'] for b in batch])
    attention_mask = torch.stack([b['attention_mask'] for b in batch])
    return {'input_ids': input_ids, 'attention_mask': attention_mask, 'answer_gt': [b['answer_gt'] for b in batch]}