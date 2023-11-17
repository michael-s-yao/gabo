"""
Plots molecule optimization results.

Author(s):
    Michael Yao @michael-s-yao

Licensed under the MIT License. Copyright University of Pennsylvania 2023.
"""
import matplotlib.pyplot as plt
import numpy as np
import argparse
import pickle


def main():
    parser = argparse.ArgumentParser(
        description="Plot molecule optimization results."
    )
    parser.add_argument(
        "results", type=str, help="Pickle results file to plot."
    )
    parser.add_argument(
        "--savepath", type=str, default=None, help="Path to save the plot to."
    )
    args = parser.parse_args()
    with open(args.results, "rb") as f:
        results = pickle.load(f)

    budget = results["batch_size"] * (
        len(results["y"]) // results["batch_size"]
    )
    idx = np.argmax(np.array(results["y"]))
    print("Surrogate:", results["y"][idx])
    print("Oracle:", results["y_gt"][idx])
    y = results["y"][:budget].reshape(-1, 16)
    y_mean, y_std = np.mean(y, axis=-1), np.std(y, axis=-1, ddof=1)
    y_gt = results["y_gt"][:budget].reshape(-1, 16)
    y_gt_mean, y_gt_std = np.mean(y_gt, axis=-1), np.std(y_gt, axis=-1, ddof=1)

    if "lipschitz" in args.results.lower():
        plt.figure(figsize=(20, 6))
        correction = results["alpha"]
        corr_mean, corr_std = [], []
        for batch in correction:
            corr_mean.append(np.mean(batch))
            corr_std.append(np.std(batch, axis=-1, ddof=1))
        corr_mean, corr_std = np.array(corr_mean), np.array(corr_std)
        plt.axhline(0.0 / (1.0 - 0.0), linestyle="--", color="k", alpha=0.5)
        plt.axhline(0.2 / (1.0 - 0.2), linestyle="--", color="k", alpha=0.5)
        plt.axhline(0.5 / (1.0 - 0.5), linestyle="--", color="k", alpha=0.5)
        plt.axhline(0.8 / (1.0 - 0.8), linestyle="--", color="k", alpha=0.5)
        plt.xlabel("Optimization Step")
        plt.xlim(0, len(correction) - 1)
        plt.ylabel(r"$\log \alpha/(1-\alpha)$")
        plt.plot(corr_mean, color="k")
        plt.fill_between(
            np.arange(len(correction)),
            corr_mean - corr_std,
            corr_mean + corr_std,
            color="k",
            alpha=0.5
        )
        plt.yscale("log")
        plt.show()
        plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(y_mean, label="Surrogate")
    plt.fill_between(
        np.arange(len(y_mean)), y_mean - y_std, y_mean + y_std, alpha=0.5
    )
    plt.plot(y_gt_mean, label="Oracle")
    plt.fill_between(
        np.arange(len(y_gt_mean)),
        y_gt_mean - y_gt_std,
        y_gt_mean + y_gt_std,
        alpha=0.5
    )
    plt.xlabel("Optimization Step")
    plt.ylabel("Penalized LogP Score")
    plt.legend()
    if args.savepath is None:
        plt.show()
    else:
        plt.savefig(
            args.savepath, transparent=True, dpi=600, bbox_inches="tight"
        )
    plt.close()


if __name__ == "__main__":
    main()
