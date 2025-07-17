from datasets import load_dataset

def _preprocess_function_map(examples, tokenizer, max_prompt_length, question_field_name, answer_field_name):
    """
    Tokenizes prompts and prepares data for the model.
    
    Args:
        examples: Dataset examples.
        tokenizer: Tokenizer instance.
        max_prompt_length: Maximum length for prompts.
        question_field_name: Field name for questions.
        answer_field_name: Field name for answers.
    
    Returns:
        Processed batch dictionary.
    """
    prompts = examples[question_field_name]
    tokenized_prompts = tokenizer(
        prompts,
        max_length=max_prompt_length,
        padding="max_length",
        truncation=True,
        return_tensors=None,
    )
    processed_batch = {
        "prompt": prompts,
        "input_ids": tokenized_prompts["input_ids"],
        "attention_mask": tokenized_prompts["attention_mask"],
        "ground_truths": examples[answer_field_name]
    }
    return processed_batch

def preprocess_dataset(dataset_name, data_dir, tokenizer, max_prompt_length, split="train"):
    """
    Loads and preprocesses a dataset.
    
    Args:
        dataset_name: Name of the dataset (e.g., "gsm8k").
        data_dir: Directory for caching data.
        tokenizer: Tokenizer instance.
        max_prompt_length: Maximum prompt length.
        split: Dataset split ("train" or "test").
    
    Returns:
        Preprocessed dataset.
    """
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
    )
    
    final_columns = ["input_ids", "attention_mask", "ground_truths", "prompt"]
    processed_dataset = processed_dataset.select_columns(final_columns)
    
    if not all(col in processed_dataset.column_names for col in final_columns):
        raise ValueError(f"Processed dataset missing required columns: {final_columns}")
    
    return processed_dataset