import torch
import matplotlib.pyplot as plt
import numpy as np  
import os
import argparse
from measure_functions import compute_attn_probs, compute_entropy

from collect_attention_maps import collect_attention_maps


def get_attention_maps(args, epoch):
    try:
        attention_maps = torch.load(f"{args.rpath}/attention_maps_{epoch}.pt")
    except Exception as e:
        attention_maps = collect_attention_maps(args, epoch)
    return attention_maps


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rpath", type=str, required=True, help="Path to experiment directory")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset to analyze (train or val)")
    parser.add_argument("--layers_to_collect", type=str, required=True, help="Comma-separated list of layer indices")
    parser.add_argument("--epochs", type=str, required=True, help="Comma-separated list of checkpoint epochs")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for analysis")
    parser.add_argument("--num_samples_to_plot", type=int, default=5, help="Number of samples to plot")
    parser.add_argument("--max_samples", type=int, default=100, help="Maximum number of samples to analyze")
    parser.add_argument("--max_size", type=int, default=1000, help="Maximum size of the dataset")
    args = parser.parse_args()

    epochs = [-1] + [int(epoch) for epoch in args.epochs.split(",")]
    num_heads = 12
    head_dim = 64

    attention_maps = collect_attention_maps(args)
    for epoch in epochs:
        print(f"Plotting epoch {epoch}")
        for layer, attn_map in attention_maps[epoch].items():
            rand_samples = np.random.randint(0, len(attn_map), size=args.num_samples_to_plot)
            attention_map = attn_map[rand_samples]
            fig, axs = plt.subplots(num_heads, 2, figsize=(10, 30), gridspec_kw={'wspace': 1})
            attn_probs = compute_attn_probs(attention_map, head_dim)
            entropy_values = compute_entropy(attn_probs)
            ranks_pre = torch.linalg.matrix_rank(attention_map)
            ranks_post = torch.linalg.matrix_rank(attn_probs)
            for sample in range(args.num_samples_to_plot):
                for head in range(num_heads):
                    axs[head, 0].imshow(attention_map[sample, head])
                    axs[head, 1].imshow(attn_probs[sample, head])
                    axs[head, 0].set_title("Rank: " + str(ranks_pre[sample, head].item()))
                    axs[head, 1].set_title("Rank: " + str(ranks_post[sample, head].item()) + "\n entropy: " + str(round(entropy_values[sample, head].item(), 2)))
            os.makedirs(f"{args.rpath}/attentions/Epoch_{epoch}/Layer_{layer}", exist_ok=True)
            plt.savefig(f"{args.rpath}/attentions/Epoch_{epoch}/Layer_{layer}/Sample_{rand_samples}.png")