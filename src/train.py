import argparse
import inspect
import json
import math
import os

import torch
import torch.distributed as dist
import tqdm

from model import save_model_and_optimizer, create_model, get_model
from data import get_dataloader
from utils import (
    setup_environment,
    save_metrics,
    MetricTracker,
    cleanup_distributed,
    get_sep_position,
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


def single_train_loop(model, optimizer, train_loader, device, ctx, args, rank, step):
    _norm = 0
    avg_token_accuracy = 0
    avg_ppl = 0
    for batch in train_loader:
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

        token_accuracy = outputs.token_accuracy.item()
        ppl = loss.exp().item()
        avg_token_accuracy += token_accuracy
        avg_ppl += ppl

        if step % 100 == 0 and rank == 0:
            # save_model_and_optimizer(model, optimizer, args, rank, step)

            reg_info = ""
            # if hasattr(get_model(model), "get_regularization_info"):
            #     reg_data = get_model(model).get_regularization_info()
            #     if reg_data["enabled"]:
            #         reg_info = f" Reg Loss: {reg_data['last_loss']:.4f}"

            # Add temperature information
            
            temp_info = ""
            # try:
            temp_data = get_model(model).get_all_temperature_info()
            if temp_data:
                mean_temps = [layer["mean_temp"] for layer in temp_data]
                avg_temp = sum(mean_temps) / len(mean_temps)
                temp_info = f" Avg Temp: {avg_temp:.3f}"
            # except:
            #     temp_info = ""

            print(
                f"Step: {step}. PPL: {ppl}. Token Accuracy: {token_accuracy}{reg_info}{temp_info}"
            )
        step += 1
    return _norm, avg_token_accuracy / step, avg_ppl / step, step


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

        _norm, token_accuracy, ppl, step = single_train_loop(
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


def create_optimizer(args, model):
    # Create Optimizer
    trainable_params = list(model.parameters())
    use_fused = "fused" in inspect.signature(torch.optim.AdamW).parameters
    extra_args = dict(fused=True) if use_fused else dict()
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, **extra_args)
    return optimizer


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

    rank, world_size, device, ptdtype, ctx = setup_environment(args)

    if rank == 0:
        print(
            f"Rank: {rank}, World Size: {world_size}, Device: {device}, PTDType: {ptdtype}, Context: {ctx}"
        )

    model, tokenizer = create_model(args, device, ptdtype, rank)
    

    # Add DDP wrapping for distributed training
    if args.distributed and world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, 
            device_ids=[device.index] if device.type == 'cuda' else None,
            find_unused_parameters=False
        )
        if rank == 0:
            print("Model wrapped with DistributedDataParallel")



    train_loader, train_sampler = get_dataloader(args, args.train_path, tokenizer, world_size, rank)
    val_loader, _ = get_dataloader(args, args.val_path, tokenizer, world_size, rank)
    if args.test_path:
        test_loader, _ = get_dataloader(args, args.test_path, tokenizer, world_size, rank)
    else:
        test_loader = None

    optimizer = create_optimizer(args, model)
    # save_model_and_optimizer(model, optimizer, args, rank, -1)

    train(
        model,
        optimizer,
        train_loader,
        val_loader,
        test_loader,
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
