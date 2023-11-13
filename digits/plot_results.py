import pickle
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Optional, Union


def plot_results(
    results: Union[Path, str],
    savepath_img: Optional[Union[Path, str]] = None,
    savepath_plt: Optional[Union[Path, str]] = None
) -> None:
    with open(results, "rb") as f:
        results = pickle.load(f)
    X, y, y_gt = results["X"], results["y"], results["y_gt"]
    batch_size = results["batch_size"]

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


if __name__ == "__main__":
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
