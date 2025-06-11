import math
import argparse
import os
import sys
import tqdm
import inspect
import logging
import random
import torch
import json
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from torch.utils.data import DataLoader
# AdamW is now used directly from torch.optim, not from transformers

from model import ImplicitModel
from configuration_model import ImplicitModelConfig
from data import CoTDataset, CoTDataCollator, extract_answer
from utils import get_sep_position, MetricTracker, setup_distributed, cleanup_distributed, get_model


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
logging.disable(logging.WARNING)


@torch.no_grad()
def evaluate(dataloader, tokenizer, device, ctx, model, keep_position=False, rank=0, world_size=1):
    model.eval()
    total_instances = 0
    total_tokens = 0
    total_correct_tokens = 0
    total_correct_answers = 0
    total_loss = 0
    position_ids_all = None
    total_correct_ans_tokens = 0
    total_ans_tokens = 0
    
    for batch in tqdm.tqdm(dataloader, disable=(rank != 0)):
        input_ids_all = batch['input_ids_all'].to(device)
        labels = batch['labels_all'].to(device)
        # Remove answer part
        sep_positions = get_sep_position(input_ids_all, tokenizer.eos_token_id)
        input_ids = input_ids_all[:, :sep_positions.max()+1]
        batch_size = input_ids.shape[0]

        with ctx:
            if keep_position:
                position_ids_all = position_ids_all[:, :input_ids_all.shape[-1]]
            outputs = get_model(model).compute_loss(input_ids=input_ids_all, labels=labels, position_ids=position_ids_all)

        total_loss += outputs.total_loss.item()
        total_correct_tokens += outputs.total_correct.item()
        total_tokens += outputs.total_tokens
        total_instances += batch_size
        total_correct_answers += outputs.total_correct_answers.item()
        total_correct_ans_tokens += outputs.correct_ans_tokens.item()
        total_ans_tokens += outputs.total_ans_tokens.item()

    # Reduce metrics across all processes
    if world_size > 1:
        metrics = torch.tensor([total_loss, total_correct_tokens, total_tokens, 
                               total_instances, total_correct_answers], 
                              dtype=torch.float32, device=device)
        dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
        total_loss, total_correct_tokens, total_tokens, total_instances, total_correct_answers = metrics.tolist()
    
    accuracy = float(total_correct_answers / total_instances)
    token_accuracy = float(total_correct_tokens / total_tokens)
    loss = float(total_loss / total_tokens)
    ppl = float(math.exp(loss))
    ans_accuracy = float(total_correct_ans_tokens / total_ans_tokens)

    return accuracy, token_accuracy, ppl, ans_accuracy



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='gpt2')
    parser.add_argument('--train_path', type=str, required=True)
    parser.add_argument('--remove_cot', action='store_true')
    parser.add_argument('--random_cot', action='store_true')
    parser.set_defaults(remove_cot=False)
    parser.set_defaults(random_cot=False)
    parser.add_argument('--val_path', type=str, required=True)
    parser.add_argument('--test_path', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--accumulate', type=int, default=1)
    parser.add_argument('--truncation', type=int, default=-1)
    parser.add_argument('--max_new_tokens', type=int, default=800)
    parser.add_argument('--max_size', type=int, default=-1) # This flag gives us the control over the size of the dataset
    parser.add_argument('--save_model', type=str, required=True)
    parser.add_argument('--save_config', type=str, required=True)
    parser.add_argument('--from_pretrained', type=str, default=None)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--bf16', action='store_true')
    parser.set_defaults(bf16=False)
    parser.add_argument('--reset_optimizer', action='store_true')
    parser.set_defaults(reset_optimizer=False)
    parser.add_argument('--keep_position', action='store_true')
    parser.set_defaults(keep_position=False)
    parser.add_argument('--reinitialize_weights', action='store_true')
    parser.set_defaults(reinitialize_weights=False)
    # Distributed training arguments
    parser.add_argument('--distributed', action='store_true', help='Enable distributed training')
    parser.set_defaults(distributed=False)
    args = parser.parse_args()


    os.makedirs(args.save_config, exist_ok=True)
    os.makedirs(args.save_model, exist_ok=True)
    json.dump(args.__dict__, open(os.path.join(args.save_config, 'args.json'), 'w'))


    # Setup distributed training
    rank, world_size, local_rank = setup_distributed()
    
    # Automatically enable distributed training if world_size > 1
    if world_size > 1:
        args.distributed = True
        if rank == 0:
            print(f"Automatically enabling distributed training (world_size={world_size})")
    
    # Only print from main process
    if rank == 0:
        print(args)
    
    # Set device based on local rank for distributed training
    if args.distributed and torch.cuda.is_available():
        device = torch.device(f'cuda:{local_rank}')
        torch.cuda.set_device(device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Set seeds for reproducibility
    random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed + rank)

    dtype = 'float32'
    if args.bf16:
        dtype = 'bfloat16'
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = torch.amp.autocast(device_type='cuda', dtype=ptdtype)
    
    if rank == 0:
        print(ptdtype, dtype, device)

    # Create model
    if args.from_pretrained is None:
        config = ImplicitModelConfig(base_model=args.model)
        model = ImplicitModel(config).to(device).to(ptdtype)
    else:
        if rank == 0:
            print(f'Loading from {args.from_pretrained}')
        model = ImplicitModel.from_pretrained(args.from_pretrained).to(device).to(ptdtype)

    model = model.to(device).to(ptdtype)
    tokenizer = model.tokenizer

    # Wrap model with DDP for distributed training
    if args.distributed and world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    #TODO: What's going on here?
    if args.reinitialize_weights:
        if rank == 0:
            print('reinitializing weights')
        underlying_model = get_model(model)
        underlying_model.base_model.apply(underlying_model.base_model._init_weights)

    # Load data
    collate_fn = CoTDataCollator(tokenizer)
    train_dataset = CoTDataset(tokenizer, args.train_path, args.truncation, max_size=args.max_size, remove_cot=args.remove_cot, random_cot=args.random_cot)
    val_dataset = CoTDataset(tokenizer, args.val_path, args.truncation, remove_cot=args.remove_cot, random_cot=False)
    
    # Use DistributedSampler for distributed training
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True) if args.distributed and world_size > 1 else None
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False) if args.distributed and world_size > 1 else None
    
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        collate_fn=collate_fn, 
        sampler=train_sampler,
        shuffle=(train_sampler is None)
    )
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        collate_fn=collate_fn, 
        sampler=val_sampler,
        shuffle=False
    )
    
    if args.test_path:
        test_dataset = CoTDataset(tokenizer, args.test_path, args.truncation)
        test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False) if args.distributed and world_size > 1 else None
        test_dataloader = DataLoader(
            test_dataset, 
            batch_size=args.batch_size, 
            collate_fn=collate_fn, 
            sampler=test_sampler,
            shuffle=False
        )
    #--------------------------------

    # Create Optimizer
    trainable_params = list(model.parameters())
    use_fused = 'fused' in inspect.signature(torch.optim.AdamW).parameters
    extra_args = dict(fused=True) if use_fused else dict()
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, **extra_args)

    # Train
    step = 0
    position_ids = None # TODO: what is that?
    best_val_accuracy = float('-inf')

    train_metric_tracker = MetricTracker()
    val_metric_tracker = MetricTracker()
    test_metric_tracker = MetricTracker()
    logits_norm_tracker = {}

    # Save the model (unwrap DDP if necessary)
    model_to_save = get_model(model)
    model_to_save.save_pretrained(os.path.join(args.save_model, f'checkpoint_-1'))

    if rank == 0:
        logits_norm_tracker[-1] = get_logits_norm(model, train_dataloader, device, ctx, position_ids, rank)

    for epoch in range(args.epochs):
        # Set epoch for DistributedSampler
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        if rank == 0:
            print(f"Epoch {epoch}.")
        model.train()
        
        _norm = 0
        for batch in tqdm.tqdm(train_dataloader, disable=(rank != 0)):
            input_ids = batch['input_ids_all'].to(device)
            labels = batch['labels_all'].to(device)
            with ctx: # this is the main part of the training loop
                if args.keep_position:
                    position_ids = position_ids[:, :input_ids.shape[-1]]
                outputs = get_model(model).compute_loss(input_ids=input_ids, labels=labels, position_ids=position_ids)
                if rank == 0:
                    _norm += outputs.logits.norm(dim=-1).sum().item()
            loss = outputs.loss
            loss.div(args.accumulate).backward()
            if step % args.accumulate == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            if step % 100 == 0 and rank == 0:
                token_accuracy = outputs.token_accuracy.item()
                ppl = loss.exp().item()
                print(f"Step: {step}. PPL: {ppl}. Token Accuracy: {token_accuracy}")
                
                
                # model_to_save = get_model(model)
                # model_to_save.save_pretrained(os.path.join(args.save_model, f'checkpoint_{step}'))
                
            step += 1

        if rank == 0:
            train_metric_tracker.update(None, token_accuracy, ppl, epoch, None)
            logits_norm_tracker[epoch] = _norm
    
        accuracy, token_accuracy, ppl, ans_token_accuracy = evaluate(val_dataloader, tokenizer, device, ctx, model, 
                                                keep_position=args.keep_position, rank=rank, world_size=world_size)
        if rank == 0:
            val_metric_tracker.update(accuracy, token_accuracy, ppl, epoch, ans_token_accuracy)
            print(f'Disable Offset Val. PPL: {ppl}; Accuracy: {accuracy}; Token Accuracy: {token_accuracy}; Ans Token Accuracy: {ans_token_accuracy}.')
            
        if accuracy > best_val_accuracy:
            if rank == 0:
                print('***best so far or removed more CoT tokens***')
            best_val_accuracy = accuracy
            if args.test_path:
                accuracy, token_accuracy, ppl, ans_token_accuracy = evaluate(test_dataloader, tokenizer, device, ctx, model, 
                                                        keep_position=args.keep_position, rank=rank, world_size=world_size)
                if rank == 0:
                    test_metric_tracker.update(accuracy, token_accuracy, ppl, epoch, ans_token_accuracy)
                    print(f'Test. PPL: {ppl}; Accuracy: {accuracy}; Token Accuracy: {token_accuracy}; Ans Token Accuracy: {ans_token_accuracy}.')
        
        # Only save from main process
        if rank == 0:
            json.dump({"token_accuracy": train_metric_tracker.token_accuracy, "ppl": train_metric_tracker.ppl}, 
                     open(os.path.join(args.save_model, 'train_metric_tracker.json'), 'w'))
            json.dump({"accuracy": val_metric_tracker.accuracy, "token_accuracy": val_metric_tracker.token_accuracy, "ppl": val_metric_tracker.ppl, "ans_token_accuracy": val_metric_tracker.ans_token_accuracy}, 
                     open(os.path.join(args.save_model, 'val_metric_tracker.json'), 'w'))
            json.dump({"accuracy": test_metric_tracker.accuracy, "token_accuracy": test_metric_tracker.token_accuracy, "ppl": test_metric_tracker.ppl, "ans_token_accuracy": test_metric_tracker.ans_token_accuracy}, 
                     open(os.path.join(args.save_model, 'test_metric_tracker.json'), 'w'))
            json.dump({"logits_norm": logits_norm_tracker}, open(os.path.join(args.save_model, 'logits_norm.json'), 'w'))
            
            # Save the model (unwrap DDP if necessary)
            model_to_save = get_model(model)
            model_to_save.save_pretrained(os.path.join(args.save_model, f'checkpoint_{epoch}'))

    # Clean up distributed training
    cleanup_distributed()


def get_logits_norm(model, loader, device, ctx, position_ids, rank=0):
    model.eval()
    with torch.no_grad():
        _norm = 0
        for batch in tqdm.tqdm(loader, disable=(rank != 0)):
            input_ids = batch['input_ids_all'].to(device)
            labels = batch['labels_all'].to(device)
            with ctx:
                outputs = get_model(model).compute_loss(input_ids=input_ids, labels=labels, position_ids=position_ids)
                _norm += outputs.logits.norm(dim=-1).sum().item()
    model.train()
    return _norm


if __name__ == "__main__":
    main()
