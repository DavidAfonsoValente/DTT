import torch
import torch.distributed as dist
import torch.optim as optim
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer
from trl.models import create_reference_model
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import wandb
import os
import sys
import yaml
import json
import gc
import argparse
from tqdm import tqdm
from copy import copy
from utils import Config, set_seed
from dataset import get_dataset, MyCollator
from DTT import DTTModel
import reward  # For PhasedReward
from custom_trainer import CustomGRPOTrainer

def main():
    parser = argparse.ArgumentParser(description="DTT Training")
    parser.add_argument("config_file", help="Path to the YAML configuration file")
    args = parser.parse_args()

    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)

    with open(args.config_file) as f:
        config_dict = yaml.safe_load(f)

    if rank == 0:
        print("Config:", config_dict)

    configs = Config(config_dict)
    set_seed(configs.seed)
    save_dir = os.path.join(configs.save_path, configs.name)

    if not os.path.exists(save_dir) and rank == 0:
        os.makedirs(save_dir)

    torch.distributed.barrier()
    cur_ckpts = os.listdir(save_dir)

    if len(cur_ckpts) > 0 and not configs.only_eval:
        if rank == 0:
            print("Warning: Found previous run; resuming from latest checkpoint. Ignoring `resume` argument!")
        checkpoints = [f for f in cur_ckpts if f.startswith("checkpoint_")]
        checkpoints.sort(key=lambda x: int(x.split("_")[1]))
        latest_checkpoint = checkpoints[-1] if checkpoints else None
        configs.resume = int(latest_checkpoint.split("_")[1])
        load_dir = os.path.join(configs.save_path, configs.name, latest_checkpoint)
        configs.load_model_path = load_dir
        print(f"Resuming from checkpoint epoch_{configs.resume}!")
    elif configs.resume != 0:
        if configs.load_model_path == "None":
            print(f"Warning: Resuming from epoch {configs.resume} without loading a checkpoint!")
        print(f"Loading from {configs.load_model_path} and skipping first {configs.resume} epochs")

    model = AutoModelForCausalLM.from_pretrained(configs.model_id)
    tokenizer = AutoTokenizer.from_pretrained(configs.model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_tokens(["<|start-latent|>", "<|end-latent|>"])
    start_latent_id = tokenizer.convert_tokens_to_ids("<|start-latent|>")
    end_latent_id = tokenizer.convert_tokens_to_ids("<|end-latent|>")
    eos_id = tokenizer.eos_token_id

    loaded = False

    if configs.load_model_path != "None":
        saved_weights = torch.load(configs.load_model_path, map_location=torch.device(rank))
        if not any(k.startswith("base_causallm") for k in saved_weights.keys()):
            loaded = True
            print(model.load_state_dict(saved_weights, strict=False))
        elif any(k.startswith("base_causallm") for k in saved_weights.keys()):
            pass
        else:
            loaded = True
            print(model.load_state_dict(saved_weights, strict=False))
        configs._name_or_path = configs.load_model_path

    model.resize_token_embeddings(len(tokenizer))
    embeddings = model.get_input_embeddings()
    target_id = tokenizer.convert_tokens_to_ids("<<")
    for token_id in [start_latent_id, end_latent_id]:
        target_embedding = embeddings.weight.data[target_id]
        embeddings.weight.data[token_id] = target_embedding
        lm_head = model.lm_head
        lm_head.weight.data[token_id] = lm_head.weight.data[target_id]

    model = DTTModel(
        base_causallm=model,
        bot_token_id=start_latent_id,
        eot_token_id=end_latent_id,
        eos_token_id=eos_id,
        tokenizer=tokenizer,
    )

    if configs.load_model_path != "None" and not loaded:
        print(model.load_state_dict(saved_weights, strict=False))

    model = model.to(rank)
    if configs.bf16:
        model.to(torch.bfloat16)

    parallel_model = DDP(model, device_ids=[rank])

    del model

    if rank == 0:
        print(parallel_model)

    base_dataset_valid = get_dataset(configs.val_path, tokenizer, max_size=32 if configs.debug else 100000000, mode='dtt')
    if not configs.only_eval:
        base_dataset_train = get_dataset(configs.train_path, tokenizer, max_size=5000 if configs.debug else 100000000, mode='dtt')

    question_val = [d["question"] for d in json.load(open(configs.val_path))]
    answers_val = [d["answer"].replace(",", "").strip() for d in json.load(open(configs.val_path))]

    if not configs.debug and rank == 0:
        wandb_run = wandb.init(project=configs.project, name=configs.name)
        wandb_run.config.update(configs, allow_val_change=True)
        text_table = wandb.Table(columns=["step", "text"])
    else:
        wandb_run = None

    if "gsm" in configs.val_path:
        max_new_tokens = 16
    else:
        max_new_tokens = 16

    total_train_steps = 0

    if not configs.only_eval:
        total_train_steps = (len(base_dataset_train) // (configs.per_device_train_batch_size * world_size)) * configs.num_train_epochs
        phased_reward = reward.PhasedReward(
            model=parallel_model.module if isinstance(parallel_model, DDP) else parallel_model,
            total_steps=total_train_steps,
            tokenizer=tokenizer,
            G=configs.num_generations,  # Use YAML value (set to 64)
            enable_binary=True,
            enable_crs=False,
            enable_lcr=False,
            enable_ede=False,
            enable_eff=True
        )

    if not configs.only_eval:
        training_args = GRPOConfig(
            output_dir=os.path.join(configs.save_path, configs.name),
            per_device_train_batch_size=configs.per_device_train_batch_size,
            num_train_epochs=configs.num_train_epochs,
            learning_rate=configs.lr,
            weight_decay=configs.weight_decay,
            gradient_accumulation_steps=configs.gradient_accumulation_steps,
            num_generations=configs.num_generations,
            beta=0.04,
            logging_steps=1,
            save_steps=configs.save_steps,
            save_strategy="steps",
            evaluation_strategy="steps",
            eval_steps=configs.eval_steps,
            max_prompt_length=512,
            max_completion_length=16,
            bf16=configs.bf16,
            use_vllm=False,
            report_to=["wandb"] if wandb_run else [],
        )

        trainer = CustomGRPOTrainer(
            model=parallel_model.module if isinstance(parallel_model, DDP) else parallel_model,
            reward_funcs=[phased_reward],  # Wrap in list as expected
            args=training_args,
            train_dataset=base_dataset_train,
            eval_dataset=base_dataset_valid,
            processing_class=tokenizer,
        )

        if wandb_run and rank == 0:
            sample_batch = next(iter(base_dataset_train))
            text_str = f"Prompt: {sample_batch['prompt']}\nAnswer: {sample_batch['answer']}"
            text_table.add_data(0, text_str)
            wandb_run.log({"data_table": copy(text_table)})

        trainer.train()
        total_train_steps = trainer.state.global_step

    collator = MyCollator(tokenizer, latent_id=start_latent_id, label_pad_token_id=-100)
    valid_dataloader = torch.utils.data.DataLoader(
        base_dataset_valid,
        num_workers=1,
        pin_memory=True,
        batch_size=1,
        collate_fn=collator,
        sampler=DistributedSampler(base_dataset_valid, shuffle=False),
    )

    if not configs.only_eval:
        valid_loss_dataloader = torch.utils.data.DataLoader(
            base_dataset_valid,
            num_workers=1,
            shuffle=False,
            pin_memory=True,
            batch_size=configs.per_device_train_batch_size,
            collate_fn=collator,
            sampler=DistributedSampler(base_dataset_valid, shuffle=False),
        )
        total_loss = 0
        with torch.no_grad():
            parallel_model.eval()
            for batch in valid_loss_dataloader:
                batch = {k: v.to(rank) for k in batch.keys() if k != "idx"}
                outputs = parallel_model(**batch)
                loss = outputs.loss
                dist.all_reduce(loss, op=dist.ReduceOp.SUM)
                total_loss += loss.item() / world_size
        if rank == 0:
            print(f"Validation loss: {total_loss / len(valid_loss_dataloader)}")
            if wandb_run:
                wandb_run.log({"eval/loss": total_loss / len(valid_loss_dataloader)})

    pbar = tqdm(
        colour="blue",
        desc="Test Accuracy",
        total=len(valid_dataloader),
        dynamic_ncols=True,
        disable=rank != 0,
    )
    cor, cor_cot, total = (
        torch.tensor(0, device=rank),
        torch.tensor(0, device=rank),
        torch.tensor(0, device=rank),
    )
    best_acc = 0.0

    with torch.no_grad():
        parallel_model.eval()
        for idx, batch in enumerate(valid_dataloader):
            test_idx = batch["idx"][0]
            batch = {k: v.to(rank) for k, v in batch.items() if k not in ["idx", "position_ids"]}
            assert len(batch["input_ids"]) == 1
            answer = answers_val[test_idx.cpu().item()]
            question = question_val[test_idx.cpu().item()]

            total += 1

            outputs = parallel_model.generate(
                **batch,
                max_new_tokens=max_new_tokens,
            )

            text_output = tokenizer.decode(outputs['sequences'][0], skip_special_tokens=True)
            answer_output = text_output.split("#")[-1].replace(",", "").strip()
            cot_output = ("\n".join(text_output.split("\n")[1:])).split("#")[0].strip()

            if idx < 5 and rank == 0:
                print(f"Question {test_idx}: Answer = '{answer}'")
                print(f"Full output: '{tokenizer.decode(outputs['sequences'][0])}'")
                print(f"Extracted Output: '{answer_output}'")

            cor += answer_output == answer

            pbar.update(1)
            pbar.set_description(f"Test accuracy: {round(float(cor / total), 2)}")

        pbar.close()
        if rank == 0:
            print(f"Device {rank}: Cor={cor}, Total={total}")

    dist.all_reduce(cor, op=dist.ReduceOp.SUM)
    dist.all_reduce(total, op=dist.ReduceOp.SUM)

    cor = cor.item()
    total = total.item()

    if rank == 0:
        print(f"Accuracy on validation set: {cor} / {total} = {cor/total}")
    sys.stdout.flush()

    if wandb_run and rank == 0:
        wandb_run.log({"eval/acc": cor / total})

    if not configs.only_eval and not configs.debug:
        if cor / total > best_acc and configs.save_only_improve:
            states = parallel_model.state_dict()
            if rank == 0:
                torch.save(states, os.path.join(save_dir, f"checkpoint_{configs.resume + 1}"))
                print("Saving model due to improved accuracy.")
            best_acc = cor / total
        elif not configs.save_only_improve and total_train_steps % configs.save_steps == 0:
            states = parallel_model.state_dict()
            if rank == 0:
                torch.save(states, os.path.join(save_dir, f"checkpoint_{configs.resume + 1}"))
                print("Saving model periodically.")
        dist.barrier()
        del states
        gc.collect()
        torch.cuda.empty_cache()

    dist.barrier()

if __name__ == "__main__":
    main()