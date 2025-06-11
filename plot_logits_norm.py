import os
import json
import numpy as np
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
for fname in os.listdir("results/5_by_5_mult"):
    if "gpt2" in fname:
        logits_norm = json.load(open(f"results/5_by_5_mult/{fname}/checkpoints/logits_norm.json"))["logits_norm"]
        if "remove_cot" in fname:
            label = "No CoT"
            if "reinit_weights" in fname:
                label += " (From Scratch)"
            else:
                label += f" (Pretrained)"
            plt.plot(list(logits_norm.keys()), list(logits_norm.values()), label=label, linestyle="--", color="red")
        else:
            label = "CoT"
            if "reinit_weights" in fname:
                label += " (From Scratch)"
            else:
                label += f" (Pretrained)"
            plt.plot(list(logits_norm.keys()), list(logits_norm.values()), label=label, color="blue")

plt.yscale("log")
plt.xlabel("Epoch")
plt.ylabel("Logits Norm")
plt.title("Logits Norm Comparison")
plt.legend()
plt.savefig("comparison_logits_norm.png")
plt.close()

