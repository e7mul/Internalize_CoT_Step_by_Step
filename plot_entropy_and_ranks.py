import torch
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import pearsonr



def plot_attention_maps(data, save_path):
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
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()



def plot_avg_entropy(data, save_path):
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
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()




def plot_avg_sink_rate(data, save_path):
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
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()




def plot_avg_frob_norm(data, save_path):
    frob_norm_mean = {}
    frob_norm_std = {}
    for e, epoch in enumerate(data):
        for layer in data[epoch]['frob_norm_metrics']:
            for metric in data[epoch]['frob_norm_metrics'][layer]:
                if 'avg_frob_norm' in metric:
                    frob_norm_mean[layer] = frob_norm_mean.get(layer, []) + [data[epoch]['frob_norm_metrics'][layer][metric]]
                elif 'std_frob_norm' in metric:
                    frob_norm_std[layer] = frob_norm_std.get(layer, []) + [data[epoch]['frob_norm_metrics'][layer][metric]]

    
    axs = plt.gca()
    for key, val in frob_norm_mean.items():
        mean = np.array(val)
        std = np.array(frob_norm_std[key])
        axs.plot(mean, label=f"Layer {key}")
        # axs.fill_between(range(len(mean)), mean - std, mean + std, alpha=0.2)
    plt.xlabel("Epoch")
    plt.ylabel("Average Frobenius Norm")
    axs.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()


def plot_entropy_frob_correlation(data, save_path):
    """
    Plot correlation between average entropy and average Frobenius norm per training step.
    
    Args:
        data: Dictionary containing the analysis data
        save_path: Path to save the correlation plot
    """
    # Extract entropy and Frobenius norm data per epoch with epoch information
    entropy_per_epoch = []
    frob_norm_per_epoch = []
    epoch_labels = []
    
    for epoch in data:
        if int(epoch.split("_")[1]) > 10:
            continue
        if epoch.split("_")[1] == "-1":
            continue
        # Calculate average entropy across all layers for this epoch
        for layer in data[epoch]['entropy_metrics']:
            entropy_per_epoch += data[epoch]['entropy_metrics'][layer]["entropies_per_head"]
            epoch_labels += [epoch]*len(data[epoch]['entropy_metrics'][layer]["entropies_per_head"])
        
        # Calculate average Frobenius norm across all layers for this epoch
        for layer in data[epoch]['frob_norm_metrics']:
            frob_norm_per_epoch += data[epoch]['frob_norm_metrics'][layer]["frob_norms_per_head"]
        

    # Convert to numpy arrays
    entropy_per_epoch = np.array(entropy_per_epoch)
    frob_norm_per_epoch = np.array(frob_norm_per_epoch)
    
    # Calculate correlation coefficient
    correlation_coef, p_value = pearsonr(entropy_per_epoch, frob_norm_per_epoch)
    
    # Create the correlation plot
    plt.figure(figsize=(10, 8))
    
    # Create color map for different epochs
    unique_epochs = list(set(epoch_labels))
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_epochs)))
    epoch_to_color = dict(zip(unique_epochs, colors))
    
    # Scatter plot with different colors for each epoch
    for epoch in unique_epochs:
        epoch_mask = [e == epoch for e in epoch_labels]
        plt.scatter(np.array(entropy_per_epoch)[epoch_mask], 
                   np.array(frob_norm_per_epoch)[epoch_mask], 
                   alpha=0.7, s=50, 
                   color=epoch_to_color[epoch], 
                   label=f'Epoch {epoch}')
    
    # Add trend line
    z = np.polyfit(entropy_per_epoch, frob_norm_per_epoch, 1)
    p = np.poly1d(z)
    plt.plot(entropy_per_epoch, p(entropy_per_epoch), "r--", alpha=0.8, linewidth=2, label='Trend line')
    
    # Add correlation information
    plt.text(0.05, 0.95, f'Correlation: {correlation_coef:.3f}\np-value: {p_value:.3e}', 
             transform=plt.gca().transAxes, fontsize=12, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    plt.xlabel('Average Entropy', fontsize=12)
    plt.ylabel('Average Frobenius Norm', fontsize=12)
    plt.title('Correlation between Average Entropy and Average Frobenius Norm per Training Step', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Save the plot
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Correlation plot saved to: {save_path}")
    print(f"Correlation coefficient: {correlation_coef:.3f}")
    print(f"P-value: {p_value:.3e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rpath", type=str, required=True)
    args = parser.parse_args()

    data = json.load(open(args.rpath + "/analysis.json"))

    plot_attention_maps(data, os.path.join(args.rpath, "plots", "attention_maps.png"))
    plot_avg_frob_norm(data, os.path.join(args.rpath, "plots", "avg_frob_norm.png"))
    plot_avg_entropy(data, os.path.join(args.rpath, "plots", "avg_entropy.png"))
    plot_avg_sink_rate(data, os.path.join(args.rpath, "plots", "avg_sink_rate.png"))

    plot_entropy_frob_correlation(data, os.path.join(args.rpath, "plots", "entropy_frob_correlation.png"))