from datasets import load_dataset

def _preprocess_function_map(examples, tokenizer, max_prompt_length, question_field_name, answer_field_name):
    """Helper function to tokenize questions and prepare them for the model."""
    prompts = examples[question_field_name]
    
    tokenized_prompts = tokenizer(
        prompts,
        max_length=max_prompt_length,
        padding="max_length", 
        truncation=True,
        return_tensors=None,
    )
    
    processed_batch = {
        "prompt": prompts,  # <-- Add the raw prompts here
        "input_ids": tokenized_prompts["input_ids"],
        "attention_mask": tokenized_prompts["attention_mask"],
        "ground_truths": examples[answer_field_name]
    }
    return processed_batch

def preprocess_dataset(dataset_name, data_dir, tokenizer, max_prompt_length, split="train"):
    """Loads and preprocesses a specified dataset."""
    if dataset_name == "gsm8k":
        raw_dataset = load_dataset("gsm8k", "main", split=split, cache_dir=data_dir)
        raw_dataset = raw_dataset.map(
            lambda x: {"processed_answer": x["answer"].split("####")[-1].strip()}
        )
        question_field, answer_field = "question", "processed_answer"
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    processed_dataset = raw_dataset.map(
        _preprocess_function_map,
        batched=True,
        fn_kwargs={
            "tokenizer": tokenizer, 
            "max_prompt_length": max_prompt_length,
            "question_field_name": question_field,
            "answer_field_name": answer_field
        },
        remove_columns=raw_dataset.column_names
    )
    
    required_columns = ["input_ids", "attention_mask", "ground_truths"]
    if not all(col in processed_dataset.column_names for col in required_columns):
        raise ValueError(f"Processed dataset is missing one or more required columns: {required_columns}")
            
    return processed_dataset
