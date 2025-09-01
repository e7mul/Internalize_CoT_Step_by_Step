import argparse
import os
import json
import torch
from typing import Dict
from measure_functions import compute_entropy, compute_sink_rate, compute_attn_probs
from utils import convert_for_json
from collect_attention_maps import collect_attention_maps


class Analysis:
    def __init__(self, attention_maps, head_dim=64):
        self.head_dim = head_dim
        self.attention_maps = attention_maps


    def analyze(self):
        results_dict = {
            "rank_metrics": self.compute_rank(self.attention_maps), 
            "entropy_metrics": self.compute_entropy(self.attention_maps), 
            "sink_rate_metrics": self.compute_sink_rate(self.attention_maps),
            "frob_norm_metrics": self.compute_frob_norm(self.attention_maps)
            }
        return results_dict


    def compute_rank(self, attention_maps: Dict) -> Dict:
        """Compute rank metrics for attention maps per head (both pre and post softmax)"""
        rank_metrics = {}
        for layer_idx, attn_maps in attention_maps.items():
            batch_size, num_heads, seq_len, _ = attn_maps.shape
            try:
                all_pre_softmax_ranks = torch.linalg.matrix_rank(attn_maps).float()
                attn_probs = compute_attn_probs(attn_maps, self.head_dim)
                all_post_softmax_ranks = torch.linalg.matrix_rank(attn_probs.float()).float()

                avg_pre_softmax_ranks_per_head = all_pre_softmax_ranks.mean(dim=0)
                avg_post_softmax_ranks_per_head = all_post_softmax_ranks.mean(dim=0)
                all_pre_ranks = all_pre_softmax_ranks.view(batch_size*num_heads)
                all_post_ranks = all_post_softmax_ranks.view(batch_size*num_heads)
            except Exception as e:
                print(f"Error computing rank for layer {layer_idx}: {e}")
                avg_pre_softmax_ranks_per_head = torch.zeros(num_heads)
                avg_post_softmax_ranks_per_head = torch.zeros(num_heads)
                all_pre_ranks = torch.zeros(batch_size*num_heads)
                all_post_ranks = torch.zeros(batch_size*num_heads)
                pass  # Skip if entropy computation fails             


            rank_metrics[layer_idx] = {
                'pre_softmax_ranks_per_head': avg_pre_softmax_ranks_per_head,
                'post_softmax_ranks_per_head': avg_post_softmax_ranks_per_head,
                'avg_pre_softmax_rank': torch.mean(all_pre_ranks).item(),
                'avg_post_softmax_rank': torch.mean(all_post_ranks).item(),
            }
        
        return rank_metrics


    def compute_entropy(self, attention_maps: Dict) -> Dict:
        """Compute entropy metrics for attention maps per head (post-softmax)"""
        entropy_metrics = {}
        for layer_idx, attn_maps in attention_maps.items():
            batch_size, num_heads, seq_len, _ = attn_maps.shape
            try:
                attn_probs = compute_attn_probs(attn_maps, self.head_dim)
                all_entropies = compute_entropy(attn_probs)
                avg_entropies_per_head = all_entropies.mean(dim=0)
                all_entropies = all_entropies.view(batch_size*num_heads)
            except Exception as e:
                print(f"Error computing entropy for layer {layer_idx}: {e}")
                avg_entropies_per_head = torch.zeros(num_heads)
                all_entropies = torch.zeros(batch_size*num_heads)
                pass  # Skip if entropy computation fails                

            entropy_metrics[layer_idx] = {
                'entropies_per_head': avg_entropies_per_head,
                'avg_entropy': torch.mean(all_entropies).item(),
                'std_entropy': torch.std(all_entropies).item(),
            }
        
        return entropy_metrics


    def compute_frob_norm(self, attention_maps: Dict) -> Dict:
        """Compute frobenius norm metrics for attention maps per head (both pre and post softmax)"""
        frob_norm_metrics = {}
        for layer_idx, attn_maps in attention_maps.items():
            batch_size, num_heads, seq_len, _ = attn_maps.shape
            try:
                frob_norms = torch.linalg.matrix_norm(attn_maps, ord=2, dim=(-2, -1))
                avg_frob_norms_per_head = frob_norms.mean(dim=0)
                all_frob_norms = frob_norms.view(batch_size*num_heads)
            except Exception as e:
                print(f"Error computing norm for layer {layer_idx}: {e}")
                avg_frob_norms_per_head = torch.zeros(num_heads)
                all_frob_norms = torch.zeros(batch_size*num_heads)

            frob_norm_metrics[layer_idx] = {
                'frob_norms_per_head': avg_frob_norms_per_head,
                'avg_frob_norm': torch.mean(all_frob_norms).item(),
                'std_frob_norm': torch.std(all_frob_norms).item(),
            }

        return frob_norm_metrics


    def compute_sink_rate(self, attention_maps: Dict) -> Dict:
        """Compute sink rate for attention maps"""
        sink_rate_metrics = {}
        for layer_idx, attn_maps in attention_maps.items():
            
            batch_size, num_heads, seq_len, _ = attn_maps.shape
            try:
                attn_probs = compute_attn_probs(attn_maps, self.head_dim)
                all_sink_rates = compute_sink_rate(attn_probs)
                avg_sink_rates_per_head = all_sink_rates.float().mean(dim=0)
                all_sink_rates = all_sink_rates.view(batch_size*num_heads).float()
            except Exception as e:
                print(f"Error computing sink rate for layer {layer_idx}: {e}")
                avg_sink_rates_per_head = torch.zeros(num_heads)
                all_sink_rates = torch.zeros(batch_size*num_heads)
                pass  # Skip if entropy computation fails                

            sink_rate_metrics[layer_idx] = {
                'sink_rates_per_head': avg_sink_rates_per_head,
                'avg_sink_rate': torch.mean(all_sink_rates).item(),
                'std_sink_rate': torch.std(all_sink_rates).item(),
            }
    
        return sink_rate_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rpath", type=str, required=True, help="Path to experiment directory")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset to analyze (train or val)")
    parser.add_argument("--layers_to_collect", type=str, required=True, help="Comma-separated list of layer indices")
    parser.add_argument("--epochs", type=str, required=True, help="Comma-separated list of checkpoint epochs")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for analysis")
    parser.add_argument("--max_samples", type=int, default=100, help="Maximum number of samples to analyze")
    parser.add_argument("--max_size", type=int, default=1000, help="Maximum size of the dataset")
    args = parser.parse_args()


    attn_maps = collect_attention_maps(args)
    
    print(list(attn_maps.keys()))
    results = {}
    # for epoch in [-1] + args.epochs.split(","):
    for epoch in args.epochs.split(","):
        analysis = Analysis(attn_maps[int(epoch)])
        results[f"epoch_{epoch}"] = analysis.analyze()

    results_serializable = convert_for_json(results)
    with open(os.path.join(args.rpath, "analysis.json"), "w") as f:
        json.dump(results_serializable, f, indent=2)
