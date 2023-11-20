import sys
import pickle
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision as thv
from pathlib import Path
from typing import Optional, Union

sys.path.append(".")
from models.metric import FID
from mnist.vae import VAE
from experiment.utility import plot_config


def plot_results(
    results: Union[Path, str],
    vae_ckpt: Union[Path, str],
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

    if "=" not in results_path.lower():
        plt.figure(figsize=(20, 6))
        plt.axhline(0.0, linestyle="--", color="k", alpha=0.5)
        plt.axhline(0.2, linestyle="--", color="k", alpha=0.5)
        plt.axhline(0.5, linestyle="--", color="k", alpha=0.5)
        plt.axhline(0.8, linestyle="--", color="k", alpha=0.5)
        plt.axhline(1.0, linestyle="--", color="k", alpha=0.5)
        plt.xlabel("Optimization Step")
        plt.xlim(0, len(results["alpha"]) - 1)
        plt.ylabel(r"$\alpha$")
        plt.plot(results["alpha"], color="k")
        if savepath_alpha is None:
            plt.show()
        else:
            plt.savefig(
                savepath_alpha, transparent=True, dpi=600, bbox_inches="tight"
            )
        plt.close()

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

    y_prog, y_gt_prog = [], []
    for i in range(1, len(y) // batch_size):
        idx = np.argmax(y[:(i * batch_size)])
        y_prog.append(y[idx])
        y_gt_prog.append(y_gt[idx])

    plt.figure(figsize=(20, 6))
    plt.plot(y_prog, label="Penalized Surrogate")
    plt.plot(y_gt_prog, label="Oracle")
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

    y, y_gt = np.squeeze(y), np.squeeze(y_gt)
    print(results_path)
    print(f"  Best Surrogate: {y_prog[-1]} | (Oracle: {y_gt_prog[-1]})")
    y_idx = np.argsort(y)
    y, y_gt = y[y_idx][-batch_size:], y_gt[y_idx][-batch_size:]
    print(
        f"  Best {batch_size} Surrogate: {np.mean(y)} +/- {np.std(y)}",
        f"| (Oracle: {np.mean(y_gt)} +/- {np.std(y_gt)})"
    )

    ref_dataset = thv.datasets.MNIST(
        "./mnist/data",
        train=False,
        download=True,
        transform=thv.transforms.ToTensor()
    )
    ref_dataset = torch.utils.data.DataLoader(
        ref_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    vae = VAE()
    vae.load_state_dict(torch.load(vae_ckpt))
    vae.eval()
    X_ref, _, _ = vae(next(iter(ref_dataset))[0])
    print(
        "  FID:",
        FID(X_ref, torch.from_numpy(X[y_idx][-batch_size:]).to(X_ref))
    )


if __name__ == "__main__":
    plot_results(
        "./mnist/docs/alpha=0.0.pkl",
        vae_ckpt="./mnist/checkpoints/mnist_vae.pt",
        savepath_img="./mnist/docs/alpha=0.0_images.png",
        savepath_plt="./mnist/docs/alpha=0.0_plot.png"
    )
    plot_results(
        "./mnist/docs/alpha=0.2.pkl",
        vae_ckpt="./mnist/checkpoints/mnist_vae.pt",
        savepath_img="./mnist/docs/alpha=0.2_images.png",
        savepath_plt="./mnist/docs/alpha=0.2_plot.png"
    )
    plot_results(
        "./mnist/docs/alpha=0.5.pkl",
        vae_ckpt="./mnist/checkpoints/mnist_vae.pt",
        savepath_img="./mnist/docs/alpha=0.5_images.png",
        savepath_plt="./mnist/docs/alpha=0.5_plot.png"
    )
    plot_results(
        "./mnist/docs/alpha=0.8.pkl",
        vae_ckpt="./mnist/checkpoints/mnist_vae.pt",
        savepath_img="./mnist/docs/alpha=0.8_images.png",
        savepath_plt="./mnist/docs/alpha=0.8_plot.png"
    )
    plot_results(
        "./mnist/docs/alpha=1.0.pkl",
        vae_ckpt="./mnist/checkpoints/mnist_vae.pt",
        savepath_img="./mnist/docs/alpha=1.0_images.png",
        savepath_plt="./mnist/docs/alpha=1.0_plot.png"
    )
    plot_results(
        "./mnist/docs/alpha.pkl",
        vae_ckpt="./mnist/checkpoints/mnist_vae.pt",
        savepath_img="./mnist/docs/alpha_images.png",
        savepath_plt="./mnist/docs/alpha_plot.png"
    )
