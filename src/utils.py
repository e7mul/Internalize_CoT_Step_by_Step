import torch
from transformers import StoppingCriteria, LogitsProcessor
import os
import json
from transformers import AutoTokenizer
import torch.distributed as dist
import numpy as np

# Convert numpy arrays and tensors to lists for JSON serialization
def convert_for_json(obj):
    if isinstance(obj, torch.Tensor):
        return obj.cpu().numpy().tolist()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_for_json(item) for item in obj]
    else:
        return obj


class MetricTracker:
    def __init__(self):
        self.accuracy = {}
        self.token_accuracy = {}
        self.ppl = {}
        self.ans_token_accuracy = {}

    def update(self, accuracy, token_accuracy, ppl, timestep, ans_token_accuracy):
        self.accuracy[timestep] = accuracy
        self.token_accuracy[timestep] = token_accuracy
        self.ppl[timestep] = ppl
        self.ans_token_accuracy[timestep] = ans_token_accuracy

    def save_as_json(self, path):
        json.dump({"accuracy": self.accuracy, "token_accuracy": self.token_accuracy, "ppl": self.ppl, "ans_token_accuracy": self.ans_token_accuracy}, open(path, 'w'))


def setup_distributed():
    """Initialize distributed training"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        rank = 0
        world_size = 1
        local_rank = 0
    
    if world_size > 1:
        print(f"[Rank {rank}] Initializing distributed training...")
        print(f"[Rank {rank}] World size: {world_size}, Local rank: {local_rank}")
        
        # Set timeout for initialization (default is 30 minutes, but we'll use 10 minutes)
        timeout = int(os.environ.get('NCCL_TIMEOUT', 600))  # 10 minutes
        
        try:
            # Initialize the process group with explicit timeout
            dist.init_process_group(
                backend='nccl', 
                rank=rank, 
                world_size=world_size,
                timeout=torch.distributed.default_pg_timeout if hasattr(torch.distributed, 'default_pg_timeout') else None
            )
            
            # Set the device for this process
            torch.cuda.set_device(local_rank)
            print(f"[Rank {rank}] Successfully initialized distributed training on GPU {local_rank}")
            
            # Test basic communication
            if world_size > 1:
                test_tensor = torch.tensor([rank], dtype=torch.float32, device=f'cuda:{local_rank}')
                dist.all_reduce(test_tensor, op=dist.ReduceOp.SUM)
                expected_sum = sum(range(world_size))
                if test_tensor.item() == expected_sum:
                    print(f"[Rank {rank}] Communication test passed!")
                else:
                    print(f"[Rank {rank}] Communication test failed! Expected {expected_sum}, got {test_tensor.item()}")
                    
        except Exception as e:
            print(f"[Rank {rank}] Failed to initialize distributed training: {e}")
            print(f"[Rank {rank}] Falling back to single-process training")
            return 0, 1, 0
    else:
        print("Single-process training mode")
    
    return rank, world_size, local_rank


def cleanup_distributed():
    """Clean up distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()


def get_model(model):
    """Get the underlying model from DDP wrapper if wrapped"""
    return model.module if hasattr(model, 'module') else model


def reduce_tensor(tensor, world_size):
    """Reduce tensor across all processes"""
    if world_size == 1:
        return tensor
    
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= world_size
    return rt

def save_model(model, tokenizer, model_dir):
    print ('saving', model_dir)
    os.makedirs(model_dir, exist_ok=True)
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)


def batch_ids(input_ids_list, pad_token_id, device, dtype):
    max_seq_len = max([len(item) for item in input_ids_list])
    batch_size = len(input_ids_list)
    input_ids = torch.Tensor(batch_size, max_seq_len).to(dtype).to(device)
    input_ids.fill_(pad_token_id)
    for batch_id in range(batch_size):
        input_ids[batch_id, :len(input_ids_list[batch_id])] = input_ids_list[batch_id]
    return input_ids


def get_sep_position(input_ids, sep_id, skip=0):
    batch_size = input_ids.shape[0]
    sep_positions = input_ids.new_zeros(batch_size).long()
    for batch_id in range(batch_size):
        mask = input_ids[batch_id].eq(sep_id)
        sep_position = mask.nonzero()[0, -1].item()
        for _ in range(skip):
            mask[sep_position] = False
            sep_position = mask.nonzero()[0, -1].item()
        sep_positions[batch_id] = sep_position
    return sep_positions


# Stop generation only after generating two EOSs, such as  z <eos> y <eos>
class DoubleEOSStoppingCriteria(StoppingCriteria):
    def __init__(self, eos_token_id):
        super().__init__()
        self.eos_token_id_ = eos_token_id
        self.init = False

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        eos_count = (input_ids == self.eos_token_id_).sum(dim=-1)
        if not self.init:
            self.init = True
            self.eos_count_init = eos_count
        done = (eos_count - self.eos_count_init) >= 2
        return done.all()

class DoubleEOSLogitsProcessor(LogitsProcessor):
    def __init__(self, eos_token_id):
        super().__init__()
        self.eos_token_id_ = eos_token_id
        self.init = False
    
    def __call__(self, input_ids, scores):
        eos_count = (input_ids == self.eos_token_id_).sum(dim=-1)
        if not self.init:
            self.init = True
            self.eos_count_init = eos_count
        done = (eos_count - self.eos_count_init) >= 2
        if done.any():
            scores[done, :] = float('-inf')
            scores[done, self.eos_token_id_] = 0
        return scores


def load_experiment_config(rpath: str):
    """Load experiment configuration from args.json file"""
    args_path = os.path.join(rpath, 'args.json')
    
    if not os.path.exists(args_path):
        raise FileNotFoundError(f"Configuration file not found: {args_path}")
    
    with open(args_path, 'r') as f:
        config_data = json.load(f)
    
    # Create a simple config object with the necessary attributes
    class ExperimentConfig:
        def __init__(self, config_data):
            # Extract relevant fields from the config
            self.model_name = config_data.get('model', 'gpt2')
            self.train_dataset = config_data.get('train_path')
            self.val_dataset = config_data.get('val_path')
            self.test_dataset = config_data.get('test_path', None)
            self.max_length = config_data.get('truncation', -1)
            self.max_size = config_data.get('max_size', -1)
            self.remove_cot = config_data.get('remove_cot', False)
            self.random_cot = config_data.get('random_cot', False)
            
            # Set up tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
    
    return ExperimentConfig(config_data)


def count_checkpoints(rpath):
    checkpoints = []
    for fname in os.listdir(rpath):
        if fname.startswith("checkpoint_"):
            checkpoints.append(int(fname.split("_")[1]))
    return sorted(checkpoints)

