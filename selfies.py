"""
Main driver program for SELFIES-related molecule generation training.

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
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from typing import Dict, Union

sys.path.append("MolOOD")
from models.vae import SELFIESVAEModule
from experiment.selfies_params import Experiment
from experiment.utility import seed_everything, plot_config
from MolOOD.molformers.datamodules.logp_dataset import LogPDataModule


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

    datamodule = LogPDataModule(
        data_root=exp.datadir,
        vocab=vocab,
        batch_size=exp.batch_size,
        num_workers=exp.num_workers
    )
    model = SELFIESVAEModule(
        in_dim=128,
        out_dim=len(datamodule.vocab.keys()),
        vocab=vocab,
        alpha=exp.alpha,
        regularization=exp.regularization,
        lr=exp.lr,
        clip=exp.clip,
        beta1=beta1,
        beta2=beta2,
        n_critic_per_generator=exp.n_critic_per_generator,
    )

    callbacks = [
        ModelCheckpoint(dirpath=exp.ckpt_dir, save_last=True)
    ]
    meta = f"molecule_alpha={exp.alpha}"
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
    if exp.find_unused_parameters and exp.alpha > 0.0:
        strategy = "ddp_find_unused_parameters_true"
    trainer = pl.Trainer(
        max_epochs=exp.num_epochs,
        accelerator=exp.device,
        logger=logger,
        callbacks=callbacks,
        deterministic=True,
        fast_dev_run=exp.fast_dev_run,
        strategy=strategy
    )

    if exp.mode in ("both", "train"):
        trainer.fit(
            model,
            datamodule=datamodule,
            ckpt_path=exp.resume_from
        )
    if exp.mode in ("both", "test"):
        try:
            model = SELFIESVAEModule.load_from_checkpoint(exp.resume_from)
        except RuntimeError:
            model = SELFIESVAEModule.load_from_checkpoint(
                exp.resume_from, map_location=torch.device("cpu")
            )
        trainer.test(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
