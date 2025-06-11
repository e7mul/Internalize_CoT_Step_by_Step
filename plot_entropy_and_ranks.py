import torch
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import os


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rpath", type=str, required=True)
    args = parser.parse_args()

    data = json.load(open(args.rpath + "/analysis.json"))


    
    for e, epoch in enumerate(data):
        fig, axs = plt.subplots(1, 2, figsize=(10, 10))
        pre_softmax = torch.tensor([])
        post_softmax = torch.tensor([])
        for layer in data[epoch]['rank_metrics']:
            for metric in data[epoch]['rank_metrics'][layer]:
                if 'pre_softmax_ranks_per_head' in metric:
                    pre_softmax = torch.cat((pre_softmax, torch.tensor(data[epoch]['rank_metrics'][layer][metric]).unsqueeze(0)), dim=0)
                elif 'post_softmax_ranks_per_head' in metric:
                    post_softmax = torch.cat((post_softmax, torch.tensor(data[epoch]['rank_metrics'][layer][metric]).unsqueeze(0)), dim=0)
    

        axs[0].imshow(pre_softmax.numpy(), vmin=0, vmax=64)
        # Add text annotations for pre_softmax
        for i in range(pre_softmax.shape[0]):
            for j in range(pre_softmax.shape[1]):
                text_color = 'white' if pre_softmax[i, j] < 13.5 else 'black'
                axs[0].text(j, i, str(int(round(pre_softmax[i, j].item()))), 
                           ha='center', va='center', color=text_color, fontsize=8)
        
        axs[1].imshow(post_softmax.numpy(), vmin=0, vmax=105)
        # Add text annotations for post_softmax
        for i in range(post_softmax.shape[0]):
            for j in range(post_softmax.shape[1]):
                text_color = 'white' if post_softmax[i, j] < 13.5 else 'black'
                axs[1].text(j, i, str(int(round(post_softmax[i, j].item()))), 
                           ha='center', va='center', color=text_color, fontsize=8)

        axs[0].set_xlabel("Head")
        axs[0].set_ylabel("Layer")
        axs[1].set_xlabel("Head")
        axs[1].set_ylabel("Layer")
        axs[0].set_title("Pre-softmax")
        axs[1].set_title("Post-softmax")
        
        os.makedirs(os.path.join(args.rpath, "plots", "attention_maps"), exist_ok=True)
        plt.savefig(os.path.join(args.rpath, "plots", "attention_maps", f"attention_maps_{epoch}.png"))
        plt.close()


    entropy_mean = {}
    entropy_std = {}
    for e, epoch in enumerate(data):
        for layer in data[epoch]['entropy_metrics']:
            for metric in data[epoch]['entropy_metrics'][layer]:
                if 'avg_entropy' in metric:
                    entropy_mean[layer] = entropy_mean.get(layer, []) + [data[epoch]['entropy_metrics'][layer][metric]]
                elif 'std_entropy' in metric:
                    entropy_std[layer] = entropy_std.get(layer, []) + [data[epoch]['entropy_metrics'][layer][metric]]


    
    axs = plt.gca()
    for key, val in entropy_mean.items():
        mean = np.array(val)
        std = np.array(entropy_std[key])
        axs.plot(mean, label=f"Layer {key}")
        axs.fill_between(range(len(mean)), np.clip(mean - std, 0, 1), mean + std, alpha=0.2)
    plt.xlabel("Epoch")
    plt.ylabel("Average Entropy")
    axs.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    os.makedirs(os.path.join(args.rpath, "plots"), exist_ok=True)
    plt.savefig(os.path.join(args.rpath, "plots", "entropy.png"))
    plt.close()



    sink_rate_mean = {}
    sink_rate_std = {}
    for e, epoch in enumerate(data):
        for layer in data[epoch]['sink_rate_metrics']:
            for metric in data[epoch]['sink_rate_metrics'][layer]:
                if 'avg_sink_rate' in metric:
                    sink_rate_mean[layer] = sink_rate_mean.get(layer, []) + [data[epoch]['sink_rate_metrics'][layer][metric]]
                elif 'std_sink_rate' in metric:
                    sink_rate_std[layer] = sink_rate_std.get(layer, []) + [data[epoch]['sink_rate_metrics'][layer][metric]]

    
    axs = plt.gca()
    for key, val in sink_rate_mean.items():
        mean = np.array(val)
        std = np.array(sink_rate_std[key])
        axs.plot(mean, label=f"Layer {key}")
        axs.fill_between(range(len(mean)), np.clip(mean - std, 0, 1), np.clip(mean + std, 0, 1), alpha=0.2)
    plt.xlabel("Epoch")
    plt.ylabel("Average Sink Rate")
    axs.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    os.makedirs(os.path.join(args.rpath, "plots"), exist_ok=True)
    plt.savefig(os.path.join(args.rpath, "plots", "sink_rate.png"))
    plt.close()
