"""
Model evaluation script and helper functions.

Author(s):
    Michael Yao

Licensed under the MIT License. Copyright University of Pennsylvania 2023.
"""
import matplotlib.pyplot as plt
import os
from pathlib import Path
import torch
from typing import Callable, Optional, Sequence, Union

from data.mnist import MNISTDataModule
from models.generator import GeneratorModule
from models.objective import Objective
from models.metric import FID
from experiment.utility import seed_everything, plot_config


def plot_func(
    models: Sequence[Union[Path, str]],
    f: Optional[Callable] = None,
    x_param: str = "alpha",
    n: int = 10_000,
    savepath: Optional[Union[Path, str]] = None,
    xlabel: Optional[str] = r"$\alpha$",
    ylabel: Optional[str] = r"$\mathrm{\mathbb{E}}[f(\theta_g(z))]$",
    seed: int = 42
) -> None:
    """
    Plots a function as a function of a specific model parameter.
    Input:
        models: list of models to plot.
        f: function to plot. Default objective function used to train the
            model.
        x_param: hyperparameter to plot along the x-axis. Default `alpha`.
        n: number of different images x to average the objective calculation
            over per model.
        xlabel: label for the x axis. Default alpha.
        ylabel: label for the y axis. Default E[f(theta_g(z))].
        seed: random seed. Default 42.
    Returns:
        None.
    """
    device = torch.device("cpu")

    X, Y = [], []
    for theta_g in models:
        theta_g = GeneratorModule.load_from_checkpoint(
            theta_g, map_location=device
        )

        if not f:
            f = Objective(
                theta_g.hparams.objective, x_dim=theta_g.hparams.x_dim
            )

        z = torch.randn((n, theta_g.hparams.z_dim)).to(device)
        X.append(getattr(theta_g.hparams, x_param))
        try:
            Y.append(torch.mean(f(theta_g(z))).item())
        except TypeError:
            dm = MNISTDataModule(batch_size=n, num_workers=0, seed=seed)
            dm.prepare_data()
            dm.setup(stage="test")
            xp, _ = next(iter(dm.test_dataloader()))
            Y.append(torch.mean(f(xp, theta_g(z))).item())

    plt.figure(figsize=(10, 6))
    plt.plot(X, Y, color="k")
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    plt.gca().set_ylim(bottom=0.0)
    if savepath:
        plt.savefig(savepath, transparent=True, bbox_inches="tight", dpi=600)
    else:
        plt.show()
    plt.close()


def make_plots(
    models: Sequence[Union[Path, str]],
    savedir: Optional[Union[Path, str]] = None,
    seed: int = 42,
) -> None:
    """
    Plots the objective function and the FID for a series of models.
    Input:
        models: list of models to plot.
        savdir: directory path to save the plots to. Default not saved.
        seed: random seed. Default 42.
    Returns:
        None.
    """
    # Plot the objective function versus alpha.
    plot_func(
        models=models,
        savepath=os.path.join(savedir, "objective_versus_alpha.png"),
        seed=seed
    )

    # Plot the FID (2-Wasserstein distance) versus alpha.
    plot_func(
        models=models,
        f=FID,
        ylabel=r"$\mathrm{FID}\left[p(x), q(x)\right]$",
        savepath=os.path.join(savedir, "fid_versus_alpha.png"),
        seed=seed
    )


def main(seed: int = 42) -> None:
    """
    Main driver function to produce plots of interest.
    Input:
        seed: random seed. Default 42.
    Returns:
        None.
    """
    seed_everything(seed=seed)
    plot_config()

    # Evaluate the models trained with 1-Wasserstein distance regularization.
    eval_set = "wasserstein/batchsize_256"
    interval_size = 0.1
    models = [
        os.path.join(
            "./ckpts", eval_set, f"energy_alpha={alpha}_epoch=199_last.ckpt"
        )
        for alpha in [
            str(round(0.1 * i, 1)) for i in range(round(1 / interval_size) + 1)
        ]
    ]
    make_plots(models, savedir=os.path.join("./docs", eval_set))

    # Evaluate the models trained with source discriminator regularization.
    eval_set = "gan_loss/batchsize_256"
    models = [
        os.path.join(
            "./ckpts", eval_set, f"energy_alpha={alpha}_epoch=199_last.ckpt"
        )
        for alpha in [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.1]
    ]
    models[2] = os.path.join(
        "./ckpts", eval_set, "energy_alpha=0.02_epoch=199_last-clip=10.ckpt"
    )
    make_plots(models, savedir=os.path.join("./docs", eval_set))


if __name__ == "__main__":
    main()
