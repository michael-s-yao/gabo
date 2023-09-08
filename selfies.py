"""
Main driver program for SELFIES-related molecule generation training.

Author(s):
    Michael Yao

Licensed under the MIT License. Copyright University of Pennsylvania 2023.
"""
import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from data.molecule import SELFIESDataModule
from models.vae import SELFIESVAEModule
from experiment.selfies_params import Experiment
from experiment.utility import seed_everything, plot_config


def main():
    exp = Experiment()
    seed_everything(seed=exp.seed)
    plot_config()
    beta1, beta2 = exp.beta

    datamodule = SELFIESDataModule(
        batch_size=exp.batch_size, num_workers=exp.num_workers, seed=exp.seed
    )
    model = SELFIESVAEModule(
        in_dim=(datamodule.max_molecule_length * len(datamodule.vocab.keys())),
        out_dim=len(datamodule.vocab.keys()),
        objective=exp.objective,
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
    meta = f"molecule_objective={exp.objective}_alpha={exp.alpha}"
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
