"""
Builds a grayscale image energy objective estimator.

Author(s):
    Michael Yao @michael-s-yao

Licensed under the MIT License. Copyright University of Pennsylvania 2023.
"""
import sys
import matplotlib.pyplot as plt
import numpy as np
import torch
import pytorch_lightning as pl
from pathlib import Path
from pytorch_lightning.callbacks import ModelCheckpoint
from typing import Optional, Union

sys.path.append(".")
from digits.surrogate import SurrogateObjective
from digits.mnist import MNISTDataModule
from experiment.utility import seed_everything, get_device


def build_surrogate_objective(
    seed: int = 42, device: torch.device = torch.device("cpu")
) -> float:
    """
    Trains an MLP regressor model as an energy objective estimator.
    Input:
        seed: random seed. Default 42.
        device: device to run model training on. Default CPU.
    Returns:
        RMSE value on the test dataset.
    """
    seed_everything(seed=seed)
    dm = MNISTDataModule(seed=seed, num_workers=0)
    surrogate = SurrogateObjective()
    callbacks = [ModelCheckpoint(monitor="val_loss", save_weights_only=True)]
    trainer = pl.Trainer(
        max_epochs=10,
        accelerator="auto",
        logger=False,
        callbacks=callbacks,
        deterministic=True,
    )
    trainer.fit(surrogate, datamodule=dm)


def eval_surrogate_objective(
    model: Union[Path, str],
    plotpath: Optional[Union[Path, str]] = None,
    device: torch.device = torch.device("cpu"),
    seed: int = 42
) -> float:
    """
    Evaluates a trained MLP regressor model as an energy objective estimator.
    Input:
        model: filepath to the trained model checkpoint to evaluate.
        plotpath: optional path to save the histogram plot to. Default None.
        device: device to run model training on. Default CPU.
        seed: random seed. Default 42.
    """
    dm = MNISTDataModule(seed=seed, num_workers=0)
    dm.prepare_data()
    dm.setup()
    model = SurrogateObjective.load_from_checkpoint(model).to(device)

    val_errors = []
    for X, _ in dm.val:
        ypred = model(X.to(device).flatten())
        y = torch.mean(torch.square(X))
        val_errors.append(ypred.item() - y.item())

    test_errors = []
    for X, _ in dm.test:
        ypred = model(X.to(device).flatten())
        y = torch.mean(torch.square(X))
        test_errors.append(ypred.item() - y.item())
    plt.figure(figsize=(10, 5))
    bins = np.linspace(-0.01, 0.01, 100)
    plt.hist(
        val_errors,
        bins=bins,
        density=True,
        label="Validation Set",
        alpha=0.7
    )
    plt.hist(
        test_errors,
        bins=bins,
        density=True,
        label="Test Set",
        alpha=0.5
    )
    plt.xlabel("Predicted Image Energy - True Image Energy")
    plt.ylabel("Probability Density")
    plt.legend()
    if plotpath is None:
        plt.show()
    else:
        plt.savefig(plotpath, dpi=600, transparent=True, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    build_surrogate_objective(device=get_device())
    eval_surrogate_objective("./digits/ckpts/surrogate.ckpt")
