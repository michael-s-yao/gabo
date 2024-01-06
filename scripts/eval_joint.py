"""
Evaluate joint VAE-surrogate models.

Author(s):
    Michael Yao

Licensed under the MIT License. Copyright University of Pennsylvania 2023.
"""
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from math import sqrt
from pathlib import Path
from torchvision.utils import make_grid
from typing import Optional, Union

import mbo  # noqa
import design_bench
from models.joint import JointVAESurrogate


def eval_joint_vae_surrogate(
    task_name: str,
    ckpt_dir: Union[Path, str],
    savepath: Optional[Union[Path, str]] = None,
    device: torch.device = torch.device("cpu"),
    num_plot: int = 8,
    seed: int = 42
) -> None:
    """
    Function to evaluate joint VAE-surrogate models.
    Input:
        task_name: name of the model-based optimization (MBO) task.
        ckpt_dir: directory with the saved model checkpoint file.
        savepath: optional path to save any function outputs to.
        device: device. Default CPU.
        num_plot: number of example images to plot (if relevant). Default 8.
        seed: random seed. Default 42.
    """
    task = design_bench.make(task_name)
    rng = np.random.RandomState(seed=seed)

    ckpt = os.path.join(ckpt_dir, task_name + ".ckpt")
    model = JointVAESurrogate.load_from_checkpoint(
        ckpt, map_location=device, task=task
    )
    print(f"Loaded model {ckpt}")
    vae, surrogate = model.vae.to(device), model.surrogate.to(device)
    dtype = next(surrogate.parameters()).dtype

    X, y = torch.from_numpy(task.x), torch.from_numpy(task.y)
    X, y = X.to(device=device, dtype=dtype), y.to(device=device, dtype=dtype)
    logits, _, _ = vae(X)
    if task_name == os.environ["MNIST_TASK"]:
        recon = torch.sigmoid(logits)
        idxs = list(range(X.size(dim=0)))
        rng.shuffle(idxs)
        idxs = idxs[:num_plot]
        grid = torch.zeros(
            (2 * num_plot, X.size(dim=-1)), device=X.device, dtype=X.dtype
        )
        for i, idx in enumerate(idxs):
            grid[2 * i], grid[(2 * i) + 1] = recon[idx], X[idx]
        grid = grid.reshape(2 * num_plot, 1, int(sqrt(X.size(dim=-1))), -1)
        grid = make_grid(grid, padding=0).permute(1, 2, 0)
        plt.imshow(grid.detach().cpu().numpy(), cmap="gray")
        plt.axis("off")
        if savepath is not None:
            plt.savefig(
                savepath, dpi=600, transparent=True, bbox_inches="tight"
            )
        else:
            plt.show()
        plt.close()
    z, _, _ = vae.encode(X)
    ypred = surrogate(z.flatten(start_dim=1))
    print("  Objective MSE:", torch.mean(torch.square(ypred - y)).item())
    if task_name not in (os.environ["BRANIN_TASK"], os.environ["MNIST_TASK"]):
        recon = torch.argmax(torch.log_softmax(logits, dim=-1), dim=-1)
        recon = recon.to(X)
        print(
            "  Accuracy:",
            (torch.sum(recon == X) / torch.numel(recon)).item()
        )
    return


def main(
    hparams_fn: Union[Path, str] = "./mbo/hparams.json",
    device: torch.device = torch.device("cpu")
):
    with open(hparams_fn, "rb") as f:
        tasks = json.load(f).keys()
    for task_name in tasks:
        eval_joint_vae_surrogate(task_name, "checkpoints", device=device)
    return


if __name__ == "__main__":
    main()
