"""
Builds a convolutional autoencoder model for MNIST digit reconstruction.

Author(s):
    Michael Yao @michael-s-yao

Licensed under the MIT License. Copyright University of Pennsylvania 2023.
"""
import math
import sys
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from pathlib import Path
from typing import Optional, Union
from pytorch_lightning.callbacks import ModelCheckpoint

sys.path.append(".")
from models.convae import ConvAutoEncLightningModule
from digits.mnist import MNISTDataModule
from experiment.utility import seed_everything


def fit(seed: int = 42):
    seed_everything(seed=seed)
    dm = MNISTDataModule(seed=seed, num_workers=0)
    model = ConvAutoEncLightningModule()

    callbacks = [ModelCheckpoint(monitor="val_mse", dirpath="./digits/ckpts")]
    trainer = pl.Trainer(
        max_epochs=100,
        accelerator="auto",
        logger=False,
        callbacks=callbacks,
        deterministic=True
    )
    trainer.fit(model, datamodule=dm)


def test(
    model: Union[Path, str],
    num_plot: int = 16,
    num_per_row: int = 4,
    seed: int = 42,
    savepath: Optional[Union[Path, str]] = None
):
    seed_everything(seed=seed)
    dm = MNISTDataModule(num_workers=0)
    model = ConvAutoEncLightningModule.load_from_checkpoint(model)
    dm.prepare_data()
    dm.setup()
    fig, axs = plt.subplots(
        math.ceil(num_plot / num_per_row), 2 * num_per_row, figsize=(20, 10)
    )
    fig.tight_layout()
    axs = axs.flatten()
    X, _ = next(iter(dm.test_dataloader()))
    for i, img in enumerate(X[:num_plot]):
        axs[(2 * i)].imshow(img[0].detach().cpu().numpy(), cmap="gray")
        axs[(2 * i)].axis("off")
        axs[(2 * i) + 1].imshow(
            model(img)[0].detach().cpu().numpy(), cmap="gray"
        )
        axs[(2 * i) + 1].axis("off")
    if savepath is None:
        plt.show()
    else:
        plt.savefig(savepath, dpi=600, transparent=True, bbox_inches="tight")
    plt.close()
    return


if __name__ == "__main__":
    fit()
    test("./digits/ckpts/convae.ckpt")
