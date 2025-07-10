import argparse
import inspect
import json
import logging
import math
import os
import random
import sys

import torch
import torch.distributed as dist
import tqdm
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from configuration_model import ImplicitModelConfig
from data import CoTDataset, CoTDataCollator, extract_answer
from model import ImplicitModel
from utils import (
    MetricTracker,
    cleanup_distributed,
    get_model,
    get_sep_position,
    setup_distributed,
)


@torch.no_grad()
def evaluate(dataloader, tokenizer, device, ctx, model, rank=0, world_size=1):
    model.eval()
    total_instances = 0
    total_tokens = 0
    total_correct_tokens = 0
    total_correct_answers = 0
    total_loss = 0
    total_correct_ans_tokens = 0
    total_ans_tokens = 0

    for batch in tqdm.tqdm(dataloader, disable=(rank != 0)):
        input_ids_all = batch["input_ids_all"].to(device)
        labels = batch["labels_all"].to(device)
        # Remove answer part
        sep_positions = get_sep_position(input_ids_all, tokenizer.eos_token_id)
        input_ids = input_ids_all[:, : sep_positions.max() + 1]
        batch_size = input_ids.shape[0]

        with ctx:
            outputs = get_model(model).compute_loss(
                input_ids=input_ids_all, labels=labels
            )

        total_loss += outputs.total_loss.item()
        total_correct_tokens += outputs.total_correct.item()
        total_tokens += outputs.total_tokens
        total_instances += batch_size
        total_correct_answers += outputs.total_correct_answers.item()
        total_correct_ans_tokens += outputs.correct_ans_tokens.item()
        total_ans_tokens += outputs.total_ans_tokens.item()

    # Reduce metrics across all processes
    if world_size > 1:
        metrics = torch.tensor(
            [
                total_loss,
                total_correct_tokens,
                total_tokens,
                total_instances,
                total_correct_answers,
            ],
            dtype=torch.float32,
            device=device,
        )
        dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
        (
            total_loss,
            total_correct_tokens,
            total_tokens,
            total_instances,
            total_correct_answers,
        ) = metrics.tolist()

    accuracy = float(total_correct_answers / total_instances)
    token_accuracy = float(total_correct_tokens / total_tokens)
    loss = float(total_loss / total_tokens)
    ppl = float(math.exp(loss))
    ans_accuracy = float(total_correct_ans_tokens / total_ans_tokens)

    return accuracy, token_accuracy, ppl, ans_accuracy


def save_model_and_optimizer(model, optimizer, args, rank, ckpt_idx):
    if rank == 0:
        print("Saving model and optimizer...")
        # Save the model (unwrap DDP if necessary)
        os.makedirs(args.save_model, exist_ok=True)
        model_to_save = get_model(model)
        model_to_save.save_pretrained(
            os.path.join(args.save_model, f"checkpoint_{ckpt_idx}")
        )
        optimizer_state_dict = optimizer.state_dict()
        torch.save(
            optimizer_state_dict,
            os.path.join(args.save_model, f"optimizer_{ckpt_idx}.pt"),
        )
        print(f"Saved model and optimizer to {args.save_model}/checkpoint_{ckpt_idx}")


def get_logits_norm(model, loader, device, ctx, rank=0):
    model.eval()
    with torch.no_grad():
        _norm = 0
        for batch in loader:
            input_ids = batch["input_ids_all"].to(device)
            labels = batch["labels_all"].to(device)
            with ctx:
                outputs = get_model(model).compute_loss(
                    input_ids=input_ids, labels=labels
                )
                _norm += outputs.logits.norm(dim=-1).sum().item()
    model.train()
    return _norm


def save_metrics(
    train_metric_tracker,
    val_metric_tracker,
    test_metric_tracker,
    logits_norm_tracker,
    args,
    rank,
):
    """Save training, validation, test metrics, and logits norm to disk (main process only)."""
    if rank != 0:
        return

    save_dir = args.save_model
    metrics_to_save = [
        (
            "train_metric_tracker.json",
            {
                "token_accuracy": train_metric_tracker.token_accuracy,
                "ppl": train_metric_tracker.ppl,
            },
        ),
        (
            "val_metric_tracker.json",
            {
                "accuracy": val_metric_tracker.accuracy,
                "token_accuracy": val_metric_tracker.token_accuracy,
                "ppl": val_metric_tracker.ppl,
                "ans_token_accuracy": val_metric_tracker.ans_token_accuracy,
            },
        ),
        (
            "test_metric_tracker.json",
            {
                "accuracy": test_metric_tracker.accuracy,
                "token_accuracy": test_metric_tracker.token_accuracy,
                "ppl": test_metric_tracker.ppl,
                "ans_token_accuracy": test_metric_tracker.ans_token_accuracy,
            },
        ),
        ("logits_norm.json", {"logits_norm": logits_norm_tracker}),
    ]

    for filename, data in metrics_to_save:
        filepath = os.path.join(save_dir, filename)
        with open(filepath, "w") as f:
            json.dump(data, f)

    print(f"Saved metrics to {save_dir}")


def single_train_loop(
    model, optimizer, train_dataloader, device, ctx, args, rank, step
):
    _norm = 0
    for batch in train_dataloader:
        input_ids = batch["input_ids_all"].to(device)
        labels = batch["labels_all"].to(device)
        with ctx:  # this is the main part of the training loop
            outputs = get_model(model).compute_loss(input_ids=input_ids, labels=labels)
            if rank == 0:
                _norm += outputs.logits.norm(dim=-1).sum().item()
        loss = outputs.loss
        loss.div(args.accumulate).backward()
        if step % args.accumulate == 0:
            # torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        if step % 100 == 0 and rank == 0:
            token_accuracy = outputs.token_accuracy.item()
            ppl = loss.exp().item()
            reg_info = ""
            if hasattr(get_model(model), "get_regularization_info"):
                reg_data = get_model(model).get_regularization_info()
                if reg_data["enabled"]:
                    reg_info = f" Reg Loss: {reg_data['last_loss']:.4f}"

            # Add temperature information
            temp_info = ""
            if args.use_temperature_scaling:
                temp_data = get_model(model).get_temperature_info()
                if temp_data:
                    mean_temps = [layer["mean_temp"] for layer in temp_data]
                    avg_temp = sum(mean_temps) / len(mean_temps)
                    temp_info = f" Avg Temp: {avg_temp:.3f}"

            print(
                f"Step: {step}. PPL: {ppl}. Token Accuracy: {token_accuracy}{reg_info}{temp_info}"
            )
        step += 1


def train(
    model,
    optimizer,
    train_dataloader,
    val_dataloader,
    test_dataloader,
    tokenizer,
    device,
    ctx,
    args,
    rank,
    world_size,
    train_sampler,
):
    step = 0
    best_val_accuracy = float("-inf")

    train_metric_tracker = MetricTracker()
    val_metric_tracker = MetricTracker()
    test_metric_tracker = MetricTracker()
    logits_norm_tracker = {}

    if rank == 0:
        logits_norm_tracker[-1] = get_logits_norm(
            model, train_dataloader, device, ctx, rank
        )

    for epoch in range(args.epochs):
        # Set epoch for DistributedSampler
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        if rank == 0:
            print(f"Epoch {epoch}.")
        model.train()

        _norm = single_train_loop(
            model, optimizer, train_dataloader, device, ctx, args, rank, step
        )

        if rank == 0:
            train_metric_tracker.update(None, token_accuracy, ppl, epoch, None)
            logits_norm_tracker[epoch] = _norm

        (
            accuracy,
            token_accuracy,
            ppl,
            ans_token_accuracy,
        ) = evaluate(
            val_dataloader,
            tokenizer,
            device,
            ctx,
            model,
            rank=rank,
            world_size=world_size,
        )
        if rank == 0:
            val_metric_tracker.update(
                accuracy, token_accuracy, ppl, epoch, ans_token_accuracy
            )
            print(
                f"Disable Offset Val. PPL: {ppl}; Accuracy: {accuracy}; Token Accuracy: {token_accuracy}; Ans Token Accuracy: {ans_token_accuracy}."
            )

        if accuracy > best_val_accuracy:
            best_val_accuracy = accuracy
            if args.test_path:
                (
                    accuracy,
                    token_accuracy,
                    ppl,
                    ans_token_accuracy,
                ) = evaluate(
                    test_dataloader,
                    tokenizer,
                    device,
                    ctx,
                    model,
                    rank=rank,
                    world_size=world_size,
                )
                if rank == 0:
                    test_metric_tracker.update(
                        accuracy, token_accuracy, ppl, epoch, ans_token_accuracy
                    )
                    print(
                        f"Test. PPL: {ppl}; Accuracy: {accuracy}; Token Accuracy: {token_accuracy}; Ans Token Accuracy: {ans_token_accuracy}."
                    )

        save_model_and_optimizer(model, optimizer, args, rank, epoch)
        save_metrics(
            train_metric_tracker,
            val_metric_tracker,
            test_metric_tracker,
            logits_norm_tracker,
            args,
            rank,
        )

    cleanup_distributed()


def load_data(args, tokenizer, world_size, rank):
    # Load data
    collate_fn = CoTDataCollator(tokenizer)
    train_dataset = CoTDataset(
        tokenizer,
        args.train_path,
        args.truncation,
        max_size=args.max_size,
        remove_cot=args.remove_cot,
        random_cot=args.random_cot,
        keep_k_target=args.keep_k_target,
    )
    val_dataset = CoTDataset(
        tokenizer,
        args.val_path,
        args.truncation,
        remove_cot=args.remove_cot,
        random_cot=False,
        keep_k_target=args.keep_k_target,
    )

    # Use DistributedSampler for distributed training
    train_sampler = (
        DistributedSampler(
            train_dataset, num_replicas=world_size, rank=rank, shuffle=True
        )
        if args.distributed and world_size > 1
        else None
    )
    val_sampler = (
        DistributedSampler(
            val_dataset, num_replicas=world_size, rank=rank, shuffle=False
        )
        if args.distributed and world_size > 1
        else None
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        sampler=val_sampler,
        shuffle=False,
    )

    test_dataloader = None
    if args.test_path:
        test_dataset = CoTDataset(tokenizer, args.test_path, args.truncation)
        test_sampler = (
            DistributedSampler(
                test_dataset, num_replicas=world_size, rank=rank, shuffle=False
            )
            if args.distributed and world_size > 1
            else None
        )
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            collate_fn=collate_fn,
            sampler=test_sampler,
            shuffle=False,
        )
    return train_dataloader, val_dataloader, test_dataloader, train_sampler


def create_model(args, device, ptdtype, rank):
    # Create model with temperature scaling options
    if args.from_pretrained is None:
        config = ImplicitModelConfig(
            base_model=args.model,
            use_temperature_scaling=args.use_temperature_scaling,
            temperature_init_value=args.temperature_init_value,
            temperature_learnable=args.temperature_learnable,
        )
        model = ImplicitModel(config).to(device).to(ptdtype)
    else:
        if rank == 0:
            print(f"Loading from {args.from_pretrained}")
        override_config = {
            "use_temperature_scaling": args.use_temperature_scaling,
            "temperature_init_value": args.temperature_init_value,
            "temperature_learnable": args.temperature_learnable,
        }
        model = (
            ImplicitModel.from_pretrained(
                args.from_pretrained, override_config=override_config
            )
            .to(device)
            .to(ptdtype)
        )

    model = model.to(device).to(ptdtype)
    tokenizer = model.tokenizer

    if args.reinitialize_weights:
        if rank == 0:
            print("reinitializing weights")
        underlying_model = get_model(model)
        underlying_model.base_model.apply(underlying_model.base_model._init_weights)

    return model, tokenizer


def create_optimizer(args, model):
    # Create Optimizer
    trainable_params = list(model.parameters())
    use_fused = "fused" in inspect.signature(torch.optim.AdamW).parameters
    extra_args = dict(fused=True) if use_fused else dict()
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, **extra_args)
    return optimizer


def setup_environment(args):
    # Setup distributed training
    rank, world_size, local_rank = setup_distributed()

    # Automatically enable distributed training if world_size > 1
    if world_size > 1:
        args.distributed = True
        if rank == 0:
            print(
                f"Automatically enabling distributed training (world_size={world_size})"
            )

    # Set device based on local rank for distributed training
    if args.distributed and torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set seeds for reproducibility
    random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed + rank)

    dtype = "float32"
    if args.bf16:
        dtype = "bfloat16"
    ptdtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[dtype]
    ctx = torch.amp.autocast(device_type="cuda", dtype=ptdtype)
    return rank, world_size, device, ptdtype, ctx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt2")
    parser.add_argument("--train_path", type=str, required=True)
    parser.add_argument("--remove_cot", action="store_true")
    parser.add_argument("--random_cot", action="store_true")
    parser.set_defaults(remove_cot=False)
    parser.set_defaults(random_cot=False)
    parser.add_argument("--val_path", type=str, required=True)
    parser.add_argument("--test_path", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--accumulate", type=int, default=1)
    parser.add_argument("--truncation", type=int, default=-1)
    parser.add_argument("--max_new_tokens", type=int, default=800)
    parser.add_argument("--max_size", type=int, default=-1)
    parser.add_argument("--save_model", type=str, required=True)
    parser.add_argument("--save_config", type=str, required=True)
    parser.add_argument("--from_pretrained", type=str, default=None)
    parser.add_argument("--seed", type=int, default=1234)
    # parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument("--bf16", action="store_true")
    parser.set_defaults(bf16=False)
    parser.add_argument("--reset_optimizer", action="store_true")
    parser.set_defaults(reset_optimizer=False)
    parser.add_argument("--reinitialize_weights", action="store_true")
    parser.set_defaults(reinitialize_weights=False)
    # Distributed training arguments
    parser.add_argument(
        "--distributed", action="store_true", help="Enable distributed training"
    )
    parser.set_defaults(distributed=False)

    # Temperature scaling arguments
    parser.add_argument(
        "--use_temperature_scaling",
        action="store_true",
        help="Use temperature-scaled attention",
    )
    parser.add_argument(
        "--temperature_init_value",
        type=float,
        default=1.0,
        help="Initial value for temperature parameters (default: 1.0)",
    )
    parser.add_argument(
        "--temperature_learnable",
        action="store_true",
        help="Make temperature parameters learnable",
    )
    parser.add_argument(
        "--reset_temperature_value",
        type=float,
        default=0.0,
        help="Rest value for temperature parameters (default: 0.0)",
    )
    parser.set_defaults(use_temperature_scaling=False)
    parser.set_defaults(temperature_learnable=False)

    parser.add_argument(
        "--keep_k_target",
        type=int,
        default=0,
        help="Keep k target tokens (default: 0)",
    )

    args = parser.parse_args()

    os.makedirs(args.save_config, exist_ok=True)
    json.dump(args.__dict__, open(os.path.join(args.save_config, "args.json"), "w"))

    if rank == 0:
        print(args)

    rank, world_size, device, ptdtype, ctx = setup_environment(args)

    if rank == 0:
        print(
            f"Rank: {rank}, World Size: {world_size}, Device: {device}, PTDType: {ptdtype}, Context: {ctx}"
        )

    model, tokenizer = create_model(args, device, ptdtype, rank)
    (
        train_dataloader,
        val_dataloader,
        test_dataloader,
        train_sampler,
    ) = load_data(args, tokenizer, world_size, rank)
    optimizer = create_optimizer(args, model)
    save_model_and_optimizer(model, optimizer, args, rank, -1)
    train(
        model,
        optimizer,
        train_dataloader,
        val_dataloader,
        test_dataloader,
        tokenizer,
        device,
        ctx,
        args,
        rank,
        world_size,
        train_sampler,
    )


if __name__ == "__main__":
    main()
