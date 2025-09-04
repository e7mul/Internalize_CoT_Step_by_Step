import os
import torch
import matplotlib.pyplot as plt
import re
import numpy as np
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rpath", type=str, required=True)
    args = parser.parse_args()

    rpath = os.path.join(args.rpath, "checkpoints")

    # Helper to extract epoch/step number from checkpoint filename
    def extract_epoch(filename):
        # Accepts both "epoch_XX.pt" and "checkpoint_XX.pt" or similar
        match = re.search(r'(?:epoch|checkpoint)[^\d]*(\d+)', filename)
        if match:
            return int(match.group(1))
        return None

    # List all checkpoint files and sort by epoch/step
    ckpt_files = [f for f in os.listdir(rpath)]
    ckpt_epochs = []
    for f in ckpt_files:
        epoch = extract_epoch(f)
        if epoch is not None:
            ckpt_epochs.append((epoch, f))
    ckpt_epochs.sort()  # sort by epoch

    if not ckpt_epochs:
        raise RuntimeError("No checkpoint files found in directory.")

    # For storing average temperature per layer per epoch
    layer_temps = {}  # layer_idx -> list of averages per epoch
    epochs = []

    for epoch, fname in ckpt_epochs:
        ckpt_path = os.path.join(rpath, fname, "state_dict.bin")
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        # Try both 'model' and direct state_dict
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        # Find all temperature_logits for each layer
        layer_avgs = {}
        for k, v in state_dict.items():
            # Look for keys like 'base_model.transformer.h.{i}.attn.temperature_logits'
            m = re.match(r'.*transformer\.h\.(\d+)\.attn\.temperature_logits', k)
            if m:
                layer_idx = int(m.group(1))
                # v is a tensor, take mean (convert to float)
                avg_temp = v.float().mean().item()
                layer_avgs[layer_idx] = avg_temp

        # Store in layer_temps
        for layer_idx, avg_temp in layer_avgs.items():
            if layer_idx not in layer_temps:
                layer_temps[layer_idx] = []
            layer_temps[layer_idx].append(avg_temp)
        epochs.append(epoch)

    # Sort layers for consistent color mapping
    sorted_layers = sorted(layer_temps.keys())

    # Pad with NaN if some epochs are missing for a layer (shouldn't happen, but for safety)
    max_len = max(len(v) for v in layer_temps.values())
    for l in sorted_layers:
        if len(layer_temps[l]) < max_len:
            layer_temps[l] += [np.nan] * (max_len - len(layer_temps[l]))

    # Plotting
    plt.figure(figsize=(10, 6))
    cmap = plt.get_cmap('tab10')
    for i, layer_idx in enumerate(sorted_layers):
        plt.plot(epochs, layer_temps[layer_idx], label=f'Layer {layer_idx}', color=cmap(i % 10), marker='o')
    plt.xlabel('Epoch/Checkpoint')
    plt.ylabel('Average Temperature')
    # plt.yscale("log")
    plt.title('Evolution of Average Temperature per Layer During Training')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(rpath, "temperature_evolution.png"), dpi=300)
