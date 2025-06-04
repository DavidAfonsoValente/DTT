import os
from datasets import load_dataset, Dataset

def preprocess_dataset(dataset_name, data_dir, split):
    """Preprocess dataset based on its type."""
    if dataset_name == "gsm8k":
        data_path = os.path.join(data_dir, "gsm8k")
        if not os.path.exists(data_path):
            os.makedirs(data_path)
            dataset = load_dataset("openai/gsm8k", "main")
            dataset.save_to_disk(data_path)
        dataset = Dataset.load_from_disk(data_path)[split]
        def process_gsm8k(examples):
            return {
                "question": examples["question"],
                "answer": [ans.split("####")[1].strip() if "####" in ans else ans for ans in examples["answer"]]
            }
        return dataset.map(process_gsm8k, batched=True)

    elif dataset_name == "prosqa":
        data_path = os.path.join(data_dir, "prosqa", f"{split}.json")
        dataset = Dataset.from_json(data_path)
        def process_prosqa(examples):
            return {
                "question": examples["question"],
                "answer": examples["answer"]
            }
        return dataset.map(process_prosqa, batched=True)

    elif dataset_name == "prontoqa":
        data_path = os.path.join(data_dir, "prontoqa")
        if not os.path.exists(data_path):
            os.makedirs(data_path)
            dataset = load_dataset("allenai/prontoqa")
            dataset.save_to_disk(data_path)
        dataset = Dataset.load_from_disk(data_path)[split]
        def process_prontoqa(examples):
            return {
                "question": examples["query"],
                "answer": examples["answer"]
            }
        return dataset.map(process_prontoqa, batched=True)

def process_answer(pred, true_answer, dataset_name):
    """Process and compare answers based on dataset type."""
    pred = pred.strip()
    true_answer = true_answer.strip()
    if dataset_name == "gsm8k":
        try:
            pred_num = float(pred)
            true_num = float(true_answer)
            return abs(pred_num - true_num) < 1e-5
        except ValueError:
            return pred == true_answer
    elif dataset_name in ["prosqa", "prontoqa"]:
        return pred.lower() == true_answer.lower()
    return False