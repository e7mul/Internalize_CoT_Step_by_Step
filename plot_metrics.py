import os
import json
import matplotlib.pyplot as plt
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rpath", type=str, required=True)
    args = parser.parse_args()

    for prefix in ["train", "val"]:
        with open(os.path.join(args.rpath, f"{prefix}_metric_tracker.json"), "r") as f:
            print(f"Loading {prefix} metric tracker from {args.rpath}")
            data = json.load(f)

            plt.plot(data["ppl"].keys(), data["ppl"].values(), label=prefix)
    plt.legend()
    plt.savefig(os.path.join(args.rpath, "ppl.png"))
    plt.show()
