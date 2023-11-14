import sys
import pickle
import matplotlib.pyplot as plt
import numpy as np
import torch
from pathlib import Path
from typing import Optional, Union

sys.path.append(".")
from digits.mnist import MNISTDataModule
from models.metric import FID
from experiment.utility import plot_config


def plot_results(
    results: Union[Path, str],
    savepath_img: Optional[Union[Path, str]] = None,
    savepath_plt: Optional[Union[Path, str]] = None,
    savepath_alpha: Optional[Union[Path, str]] = None
) -> None:
    plot_config()
    results_path = results
    with open(results, "rb") as f:
        results = pickle.load(f)
    X, y, y_gt = results["X"], results["y"], results["y_gt"]
    batch_size = results["batch_size"]

    if "lipschitz" in results_path.lower():
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
        plt.ylabel(r"$\alpha/(1-\alpha)$")
        plt.plot(corr_mean, color="k")
        plt.fill_between(
            np.arange(len(correction)),
            corr_mean - corr_std,
            corr_mean + corr_std,
            color="k",
            alpha=0.5
        )
        if savepath_alpha is None:
            plt.show()
        else:
            plt.savefig(
                savepath_alpha, transparent=True, dpi=600, bbox_inches="tight"
            )
        plt.close()
        sys.exit(0)

    num_images = 5
    idxs = np.linspace(0, X.shape[0] // batch_size, num=num_images, dtype=int)
    idxs[0] += 1
    fig, ax = plt.subplots(1, num_images, figsize=(20, 4))
    ax = ax.flatten()
    for i, bound in enumerate(idxs):
        ax[i].imshow(
            X[np.argmax(y[:(bound * batch_size)]), 0, :, :], cmap="gray"
        )
        ax[i].axis("off")
        ax[i].set_title(bound)
    if savepath_img is None:
        plt.show()
    else:
        plt.savefig(
            savepath_img, dpi=600, transparent=True, bbox_inches="tight"
        )
    plt.close()

    y, y_gt = y.reshape(-1, batch_size), y_gt.reshape(-1, batch_size)
    y_mean, y_std = np.mean(y, axis=-1), np.std(y, axis=-1, ddof=1)
    y_gt_mean, y_gt_std = np.mean(y_gt, axis=-1), np.std(y_gt, axis=-1, ddof=1)

    best_idx = 0
    for i in range(len(y_mean)):
        if y_mean[i] > y_mean[best_idx]:
            best_idx = i
            continue
        y_mean[i], y_std[i] = y_mean[best_idx], y_std[best_idx]
        y_gt_mean[i], y_gt_std[i] = y_gt_mean[best_idx], y_gt_std[best_idx]

    plt.figure(figsize=(20, 6))
    plt.plot(y_mean, label="Penalized Surrogate")
    plt.fill_between(
        np.arange(X.shape[0] // batch_size),
        y_mean - y_std,
        y_mean + y_std,
        alpha=0.5
    )
    plt.plot(y_gt_mean, label="Oracle")
    plt.fill_between(
        np.arange(X.shape[0] // batch_size),
        y_gt_mean - y_gt_std,
        y_gt_mean + y_gt_std,
        alpha=0.5
    )
    plt.xlabel("Optimization Step")
    plt.xlim(0, X.shape[0] // batch_size - 1)
    plt.ylabel(r"Objective $y=||x||_2^2$")
    plt.legend()
    if savepath_plt is None:
        plt.show()
    else:
        plt.savefig(
            savepath_plt, dpi=600, transparent=True, bbox_inches="tight"
        )
    plt.close()

    dm = MNISTDataModule()
    dm.prepare_data()
    dm.setup()
    print(results_path)
    print("  Surrogate:", y_mean[-1], y_std[-1])
    print("  Oracle:", y_gt_mean[-1], y_gt_std[-1])
    fid = FID(
        next(iter(dm.test_dataloader()))[0][-(4 * batch_size):],
        torch.from_numpy(X[-(4 * batch_size):])
    )
    print("  FID:", fid.item())


if __name__ == "__main__":
    plot_results(
        "./digits/docs/alpha=lipschitz.pkl",
        savepath_img="./digits/docs/alpha=lipschitz_images.png",
        savepath_plt="./digits/docs/alpha=lipschitz_plot.png",
        savepath_alpha="./digits/docs/alpha=lipschitz_alpha.png"
    )
    sys.exit(0)
    plot_results(
        "./digits/docs/alpha=0.0.pkl",
        savepath_img="./digits/docs/alpha=0.0_images.png",
        savepath_plt="./digits/docs/alpha=0.0_plot.png"
    )
    plot_results(
        "./digits/docs/alpha=0.2.pkl",
        savepath_img="./digits/docs/alpha=0.2_images.png",
        savepath_plt="./digits/docs/alpha=0.2_plot.png"
    )
    plot_results(
        "./digits/docs/alpha=0.5.pkl",
        savepath_img="./digits/docs/alpha=0.5_images.png",
        savepath_plt="./digits/docs/alpha=0.5_plot.png"
    )
    plot_results(
        "./digits/docs/alpha=0.8.pkl",
        savepath_img="./digits/docs/alpha=0.8_images.png",
        savepath_plt="./digits/docs/alpha=0.8_plot.png"
    )
    plot_results(
        "./digits/docs/alpha=1.0.pkl",
        savepath_img="./digits/docs/alpha=1.0_images.png",
        savepath_plt="./digits/docs/alpha=1.0_plot.png"
    )
    plot_results(
        "./digits/docs/alpha=lipschitz.pkl",
        savepath_img="./digits/docs/alpha=lipschitz_images.png",
        savepath_plt="./digits/docs/alpha=lipschitz_plot.png"
    )
