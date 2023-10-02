"""
Main driver program for molecule generation training.

Author(s):
    Michael Yao

Licensed under the MIT License. Copyright University of Pennsylvania 2023.
"""
import os
import json
import sys
from pathlib import Path
import torch
import lightning.pytorch as pl
from lightning.pytorch.strategies.ddp import DDPStrategy
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from typing import Dict, Union

sys.path.append("MolOOD")
from data.molecule import SELFIESDataModule
from models.molgenerator import MolGeneratorModule
from experiment.selfies_params import Experiment
from experiment.utility import seed_everything, plot_config


def load_vocab(fn: Union[Path, str]) -> Dict[str, int]:
    """
    Loads the molecule vocabulary dict mapping molecular components to
    numbers.
    Input:
        fn: File path to the vocab dictionary.
    Returns:
        A dictionary mapping molecular components to numbers.
    """
    with open(fn, "rb") as f:
        vocab = json.load(f)

    if "[start]" not in vocab:
        raise ValueError("Vocab must contain `[start]` token.")
    if "[stop]" not in vocab:
        raise ValueError("Vocab must contain `[stop]` token.")
    if "[pad]" not in vocab:
        raise ValueError("Vocab must contain `[pad]` token.")

    return vocab


def main():
    exp = Experiment()
    seed_everything(seed=exp.seed)
    plot_config()
    beta1, beta2 = exp.beta

    vocab = load_vocab(os.path.join(exp.datadir, "vocab.json"))

    datamodule = SELFIESDataModule(
        vocab=vocab,
        root=exp.datadir,
        batch_size=exp.batch_size,
        num_workers=exp.num_workers,
        seed=exp.seed
    )
    model = MolGeneratorModule(
        vocab,
        max_molecule_length=datamodule.max_molecule_length,
        alpha=exp.alpha,
        lr=exp.lr,
        optimizer=exp.optimizer,
        beta=exp.beta,
        weight_decay=exp.weight_decay
    )

    meta = f"molecule_alpha={exp.alpha}"
    callbacks = [
        ModelCheckpoint(
            dirpath=exp.ckpt_dir,
            monitor="val_loss",
            mode="min",
            filename=f"{meta}_{{epoch}}",
            save_last=True
        ),
        EarlyStopping(
            monitor="val_loss",
            min_delta=0.0,
            patience=20,
            verbose=False,
            mode="min"
        )
    ]
    callbacks[0].CHECKPOINT_NAME_LAST = f"{meta}_{{epoch}}_last"

    logger = False
    if (
        not exp.disable_wandb and
        not exp.fast_dev_run and
        not exp.mode == "test"
    ):
        logger = WandbLogger(
            project=os.path.basename(os.path.dirname(__file__)),
            log_model="all",
        )
        logger.log_hyperparams(exp.args)

    strategy = "auto"
    if exp.find_unused_parameters:
        strategy = DDPStrategy(find_unused_parameters=True)
    trainer = pl.Trainer(
        max_epochs=exp.num_epochs,
        accelerator=exp.device,
        logger=logger,
        callbacks=callbacks,
        deterministic=True,
        fast_dev_run=exp.fast_dev_run,
        strategy=strategy,
        devices=exp.devices
    )

    if exp.mode in ("both", "train"):
        trainer.fit(
            model,
            datamodule=datamodule,
            ckpt_path=exp.resume_from
        )
    if exp.mode in ("both", "test"):
        try:
            if exp.model.lower() == "molgan":
                model = MolGANModule.load_from_checkpoint(
                    exp.resume_from
                )
            elif exp.model.lower() == "vae":
                model = SELFIESVAEModule.load_from_checkpoint(exp.resume_from)
        except RuntimeError:
            if exp.model.lower() == "molgan":
                model = MolGANModule.load_from_checkpoint(
                    exp.resume_from, map_location=torch.device("cpu")
                )
            elif exp.model.lower() == "vae":
                model = SELFIESVAEModule.load_from_checkpoint(
                    exp.resume_from, map_location=torch.device("cpu")
                )
        trainer.test(model, datamodule=datamodule)


if __name__ == "__main__":
    torch.set_float32_matmul_precision("medium")
    main()
