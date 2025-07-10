import json
import matplotlib.pyplot as plt
import numpy as np
import torch

fig, axs = plt.subplots(1, 1, figsize=(6, 4))

rpaths = ["results/test_gdrive/analysis.json", "results/5_by_5_mult/gpt2_20250608_114605_remove_cot/analysis.json"]

for e, rpath in enumerate(rpaths):
    with open(rpath, "r") as f:
        if e == 0:
            data = json.load(f)["epoch_0"]["entropy_metrics"]
            label = "Distilled"
        else:
            data = json.load(f)["epoch_39"]["entropy_metrics"]
            label = "Baseline"
        xticks = list(data.keys())
        values = np.array([data[k]["avg_entropy"] for k in xticks])
        std_values = np.array([data[k]["std_entropy"] for k in xticks])
        axs.plot(xticks, values, marker="o", label=label)
        axs.fill_between(xticks, values - std_values, values + std_values, alpha=0.2)

axs.legend()
axs.set_xlabel("Layer")
axs.set_ylabel("Entropy")
axs.set_title("Entropy per Layer")
axs.grid(True)
plt.savefig("entropy_per_layer.png", dpi=300)




for epoch in [-1, 0, 10, 20, 30, 39]:
    fig, axs = plt.subplots(1, 1, figsize=(6, 4))

    rpaths = ["results/test_gdrive/analysis.json", "results/5_by_5_mult/gpt2_20250608_114605_remove_cot/analysis.json"] #, "results/5_by_5_mult/gpt2_seqvcr_pause_20250612_135246/analysis.json"]

    for e, rpath in enumerate(rpaths):
        with open(rpath, "r") as f:
            if e == 0:
                data = json.load(f)["epoch_0"]["entropy_metrics"]
                label = "Distilled"
                color = "blue"
            elif e == 1:
                data = json.load(f)[f"epoch_{epoch}"]["entropy_metrics"]
                label = "Baseline"
                color = "orange"
            elif e == 2:
                data = json.load(f)["epoch_39"]["entropy_metrics"]
                label = "SeqVCR"
                color = "green"
            xticks = list(data.keys())
            values = np.array([data[k]["avg_entropy"] for k in xticks])
            axs.plot(xticks, values, marker="o", label=label, color=color, markersize=4, alpha=0.5)
            values = torch.cat([torch.tensor(data[k]["entropies_per_head"]).unsqueeze(0) for k in xticks])
            print(values.shape)

            for i in range(values.shape[1]):
                axs.scatter(xticks, values[:, i], color=color, s=15, alpha=0.5, marker='x')

    axs.legend()
    axs.set_xlabel("Layer")
    axs.set_ylabel("Entropy")
    axs.set_title("Entropy per Layer")
    axs.grid(True)
    plt.savefig(f"entropy_per_layer_detailed_{epoch}.png", dpi=300)

