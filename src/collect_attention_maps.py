import os
import torch
from torch.utils.data import DataLoader
from typing import Dict, List
import tqdm

from utils import load_experiment_config, count_checkpoints
from model import ImplicitModel
from data import CoTDataset, CoTDataCollator
from configuration_model import ImplicitModelConfig


class CollectAttentionMaps:
    def __init__(self, model, tokenizer, dataset, device=None, batch_size=1, max_samples=100):
        self.model = model
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size
        self.max_samples = max_samples

    def collect_attention_maps(self, layers_to_collect: List[int]) -> Dict:
        """Collect pre-softmax attention maps using forward hooks on attention modules"""
        self.model.eval()
        
        # Set up data loader
        collate_fn = CoTDataCollator(self.tokenizer)
        dataloader = DataLoader(
            self.dataset, 
            batch_size=self.batch_size,
            collate_fn=collate_fn,
            shuffle=True
        )
        
        attention_maps = {layer: [] for layer in layers_to_collect}
        hooks = []
        layer_outputs = {}
        
        def attention_hook(layer_idx):
            def hook(module, input, output):
                layer_outputs[layer_idx] = input[0].detach().cpu()
            return hook
        
        def compute_attention(module, hidden_states, layer_idx):
            try:
                bsz, seq_len = hidden_states.size()[:2]
                query, key, value = module.c_attn(hidden_states).split(module.split_size, dim=2)
                
                num_heads = module.num_heads
                head_dim = module.head_dim
                
                query = query.view(bsz, seq_len, num_heads, head_dim).transpose(1, 2)
                key = key.view(bsz, seq_len, num_heads, head_dim).transpose(1, 2)
                
                attn_weights = torch.matmul(query, key.transpose(-1, -2))
                attention_maps[layer_idx].append(attn_weights.detach().cpu())
                return True
            except Exception as e:
                print(f"Error computing attention for layer {layer_idx}: {e}")
                return False
        
        try:
            model_to_hook = self.model.module if hasattr(self.model, 'module') else self.model
            
            if hasattr(model_to_hook.base_model, 'transformer'):
                for layer_idx in layers_to_collect:
                    if layer_idx < len(model_to_hook.base_model.transformer.h):
                        attn_module = model_to_hook.base_model.transformer.h[layer_idx].attn
                        hook = attn_module.register_forward_hook(attention_hook(layer_idx))
                        hooks.append((layer_idx, hook, attn_module))
            
            collected_samples = 0
            with torch.no_grad():
                for batch in tqdm.tqdm(dataloader):
                    if collected_samples >= self.max_samples:
                        break
                    collected_samples += len(batch['input_ids_all'])
                    
                    input_ids = batch['input_ids_all'].to(self.device)
                    layer_outputs.clear()
                    
                    try:
                        self.model.forward(input_ids=input_ids, output_attentions=False)
                        
                        for layer_idx, hook, attn_module in hooks:
                            if layer_idx in layer_outputs:
                                hidden_states = layer_outputs[layer_idx].to(self.device)
                                compute_attention(attn_module, hidden_states, layer_idx)
                    except Exception as e:
                        print(f"Error during forward pass: {e}")
                        continue
        
        finally:
            for layer_idx, hook, attn_module in hooks:
                hook.remove()

        for layer_idx, maps in attention_maps.items():
            attention_maps[layer_idx] = torch.cat(maps, dim=0)[:self.max_samples]
        
        return attention_maps


def collect_attention_maps(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.layers_to_collect == "all":
        layers_to_collect = list(range(12))
    else:
        layers_to_collect = [int(x.strip()) for x in args.layers_to_collect.split(',')]
    

    if args.epochs == "all":
        epochs_to_analyze = count_checkpoints(os.path.join(args.rpath, "checkpoints"))
    else:
        epochs_to_analyze = [int(x.strip()) for x in args.epochs.split(',')]

    config = load_experiment_config(args.rpath)

    if args.dataset == "train":
        dataset = CoTDataset(config.tokenizer, config.train_dataset, config.max_length, args.max_size, config.remove_cot, config.random_cot)
    elif args.dataset == "val":
        dataset = CoTDataset(config.tokenizer, config.val_dataset, config.max_length, args.max_size, config.remove_cot, config.random_cot)

    # Create base model
    model_config = ImplicitModelConfig(base_model=config.model_name)
    model = ImplicitModel(model_config).to(device)

    print(model)

    # Analyze each specified checkpoint
    results = {}
    for epoch in [-1] + epochs_to_analyze:
        ckpt_path = os.path.join(os.path.join(args.rpath, "checkpoints"), f"checkpoint_{epoch}", "state_dict.bin")
        
        if not os.path.exists(ckpt_path):
            print(f"Warning: Checkpoint {ckpt_path} not found, skipping...")
            continue
        
        # Load checkpoint
        state_dict = torch.load(ckpt_path, map_location=device)
        model_to_load = model.module if hasattr(model, 'module') else model
        model_to_load.load_state_dict(state_dict, strict=True)

        # Run analysis
        
        attention_maps = CollectAttentionMaps(model, config.tokenizer, dataset, device, args.batch_size, args.max_samples).collect_attention_maps(layers_to_collect)
        results[epoch] = attention_maps

    return results
