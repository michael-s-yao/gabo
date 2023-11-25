"""
Analyze results from MNIST-related training.

Author(s):
    Michael Yao

Licensed under the MIT License. Copyright University of Pennsylvania 2023.
"""
import sys
import os
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


def main(
    results_dir: Union[Path, str] = "mnist/docs/final",
    savepath: Optional[Union[Path, str]] = None
):
    directory, _, fns = next(os.walk(results_dir))
    fns = [os.path.join(directory, f) for f in fns]
    alphas = {a: [] for a in ["0.0", "0.2", "0.5", "0.8", "1.0", "Ours"]}
    for f in fns:
        key = f.split("=")[1].split("_")[0]
        if "pkl" in key:
            alphas["Ours"].append(f)
        elif key in alphas.keys():
            alphas[key].append(f)

    print("     |    Surrogate   |     Oracle")
    print("----------------------------------------")
    for a, fns in alphas.items():
        surr, best = [], []
        for pkl in fns:
            with open(pkl, "rb") as f:
                results = pickle.load(f)
            y, y_gt = np.squeeze(results["y"]), np.squeeze(results["y_gt"])
            idxs = np.argsort(y)
            surr += [y[idxs][-1]]
            best += [y_gt[idxs][-1]]
        best = np.array(best)

        padding = " " if np.mean(surr) > 0.0 else ""
        a = a + " " if a != "Ours" else a
        print(
            f"{a} | {padding}{np.mean(surr):.2f} +/- {np.std(surr):.2f} |",
            f"{np.mean(best):.2f} +/- {np.std(best):.2f}"
        )

    a = []
    for pkl in alphas["Ours"]:
        with open(pkl, "rb") as f:
            results = pickle.load(f)
        a += results["alpha"].tolist()
    a = np.array(a).reshape(len(alphas["Ours"]), -1)
    plt.figure(figsize=(10, 5))
    plt.plot(1 + np.arange(a.shape[-1]), np.mean(a, axis=0), color="k")
    plt.xlabel("Optimization Step")
    plt.ylabel(r"$\alpha$")
    plt.ylim(0.0, 1.0)
    if savepath is None:
        plt.show()
    else:
        plt.savefig(savepath, dpi=600, bbox_inches="tight", transparent=True)
    plt.close()


if __name__ == "__main__":
    main()
