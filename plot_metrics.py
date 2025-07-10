import os
import json
import matplotlib.pyplot as plt
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rpath", type=str, required=True)
    args = parser.parse_args()

    plt.figure(figsize=(10, 5))
    for metric in ["ppl", "token_accuracy"]:
        plt.gca().clear()
        for prefix in ["train", "val"]:
            if prefix == "val" and metric == "token_accuracy":
                metric = "ans_token_accuracy"
            with open(os.path.join(args.rpath, "checkpoints", f"{prefix}_metric_tracker.json"), "r") as f:
                print(f"Loading {prefix} metric tracker from {args.rpath}")
                data = json.load(f)
                plt.plot(data[metric].keys(), data[metric].values(), label=prefix)
        plt.legend()
        plt.savefig(os.path.join(args.rpath, f"{metric}.png"))
        plt.show()
