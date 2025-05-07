"""
Dataset Utilities for DTT - dataset.py

This module provides dataset loading and processing for both supervised CoT and unsupervised RL DTT.
"""

import json
import itertools
import random
from dataclasses import dataclass
from typing import Optional

import torch
import torch.distributed as dist
from datasets import Dataset
from transformers import PreTrainedTokenizerBase
from transformers.data.data_collator import pad_without_fast_tokenizer_warning

def get_dataset(path, tokenizer, max_size=1000000000, mode='cot'):
    """
    Loads and processes the dataset based on the specified mode.
    
    Args:
        path (str): Path to the JSON dataset.
        tokenizer: Tokenizer object.
        max_size (int): Maximum number of samples to load.
        mode (str): 'cot' for supervised CoT, 'dtt' for unsupervised DTT RL.
    
    Returns:
        Dataset: Processed dataset with fields depending on mode.
    """
    def tokenize_sample_cot(sample):
        question_tokenized = tokenizer.encode(
            sample["question"] + "\n", add_special_tokens=True
        )
        steps_tokenized = [
            tokenizer.encode(s + "\n", add_special_tokens=False)
            for s in sample["steps"]
        ]
        answer_tokenized = tokenizer.encode(
            "### " + sample["answer"], add_special_tokens=False
        ) + [tokenizer.eos_token_id]
        return {
            "question_tokenized": question_tokenized,
            "steps_tokenized": steps_tokenized,
            "answer_tokenized": answer_tokenized,
            "idx": sample["idx"],
        }
    
    def tokenize_sample_dtt(sample):
        question_tokenized = tokenizer.encode(
            sample["question"] + "\n", add_special_tokens=True
        )
        return {
            "question_tokenized": question_tokenized,
            "answer": sample["answer"],
            "idx": sample["idx"],
        }
    
    data = json.load(open(path))[:max_size]
    data = [{**d, "idx": idx} for idx, d in enumerate(data)]
    keys = data[0].keys()
    dataset = Dataset.from_dict({k: [d[k] for d in data] for k in keys})
    
    tokenize_sample = tokenize_sample_cot if mode == 'cot' else tokenize_sample_dtt
    
    if torch.cuda.device_count() > 1:
        if dist.get_rank() == 0:
            processed_dataset = [
                dataset.map(
                    tokenize_sample, remove_columns=list(dataset.features), num_proc=32
                )
            ]
        else:
            processed_dataset = [None]
        dist.broadcast_object_list(processed_dataset, src=0)
        dataset = processed_dataset[0]
    else:
        dataset = dataset.map(
            tokenize_sample, remove_columns=list(dataset.features), num_proc=32
        )
    
    if mode == 'cot':
        d = data[0]
        complete = d["question"] + "\n" + "\n".join(d["steps"]) + "\n### " + d["answer"]
        complete_tokenized = tokenizer.encode(complete, add_special_tokens=True) + [
            tokenizer.eos_token_id
        ]
        assert (
            complete_tokenized
            == dataset[0]["question_tokenized"]
            + list(itertools.chain.from_iterable(dataset[0]["steps_tokenized"]))
            + dataset[0]["answer_tokenized"]
        )
    
    return dataset

@dataclass
class MyCollator:
    tokenizer: PreTrainedTokenizerBase
    latent_id: Optional[int] = None
    label_pad_token_id: Optional[int] = -100
    
    def __call__(self, features, return_tensors=None):
        assert self.tokenizer.padding_side == "right"
        
        if "labels" in features[0].keys():
            label_name = "labels"
        elif "label" in features[0].keys():
            label_name = "label"
        else:
            label_name = None
        
        non_label_position_features = [
            {
                k: v
                for k, v in feature.items()
                if k != label_name and k != "position_ids"
            }
            for feature in features
        ]
        
        return_tensors = "pt"
        
        batch = pad_without_fast_tokenizer_warning(
            self.tokenizer,
            non_label_position_features,
            padding=True,
            pad_to_multiple_of=None,
            return_tensors=return_tensors,
        )
        
        if label_name:
            labels = [feature[label_name] for feature in features]
            if all(label is None for label in labels):
                labels = None
            else:
                max_label_length = max(len(l) for l in labels)
                batch["labels"] = [
                    label + [self.label_pad_token_id] * (max_label_length - len(label))
                    for label in labels
                ]
                batch["labels"] = torch.tensor(batch["labels"], dtype=torch.int64)
        
        if "position_ids" in features[0].keys():
            position_ids = [feature["position_ids"] for feature in features]
            max_pos_length = max(len(l) for l in position_ids)
            batch["position_ids"] = [
                pos + [0] * (max_pos_length - len(pos))
                for pos in position_ids
            ]
            batch["position_ids"] = torch.tensor(batch["position_ids"], dtype=torch.int64)
        
        return batch

def get_question_latent_dataset(
    scheduled_stage,
    base_dataset_valid,
    configs,
    start_id,
    latent_id,
    end_id,
    no_special_marker=False,
):
    def process_dataset(sample):
        tokens = (
            sample["question_tokenized"]
            + ([] if no_special_marker else [start_id])
            + ([] if no_special_marker else [end_id])
        )
        return {
            "input_ids": tokens,
            "idx": sample["idx"],
            "attention_mask": [1] * len(tokens),
            "position_ids": list(range(len(tokens))),
        }
    
    return base_dataset_valid.map(
        process_dataset, remove_columns=list(base_dataset_valid.features), num_proc=32
    )

def get_cot_latent_dataset(
    scheduled_stage,
    base_dataset,
    configs,
    start_id,
    latent_id,
    end_id,
    no_special_marker=False,
    shuffle=False,
):
    def process_dataset(sample):
        tokens = sample["question_tokenized"] + [start_id]
        return {
            "input_ids": tokens,
            "attention_mask": [1] * len(tokens),
            "answer": sample["answer"],
            "idx": sample["idx"],
        }
    
    dataset = base_dataset.map(
        process_dataset, remove_columns=list(base_dataset.features), num_proc=32
    )
    if shuffle:
        dataset = dataset.shuffle()
    return dataset