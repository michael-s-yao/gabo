"""
Builds a grayscale image energy objective estimator.

Author(s):
    Michael Yao @michael-s-yao

Licensed under the MIT License. Copyright University of Pennsylvania 2023.
"""
import sys
import torch
import pytorch_lightning as pl
from pathlib import Path
from pytorch_lightning.callbacks import ModelCheckpoint
from typing import Optional, Union

sys.path.append(".")
from digits.surrogate import SurrogateObjective
from data.mnist import MNISTDataModule
from experiment.utility import seed_everything, get_device


def build_surrogate_objective(
    seed: int = 42,
    plotpath: Optional[Union[Path, str]] = None,
    savepath: Optional[Union[Path, str]] = None,
    device: torch.device = torch.device("cuda")
) -> float:
    """
    Trains and tests an MLP regressor model as a energy objective estimator.
    Input:
        seed: random seed. Default 42.
        plotpath: optional path to save the histogram plot to. Default None.
        savepath: optional path to save the model to. Default None.
        device: device to run model training on. Default GPU.
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
    trainer.test(surrogate, datamodule=dm, ckpt_path="best")


if __name__ == "__main__":
    build_surrogate_objective(device=get_device())
