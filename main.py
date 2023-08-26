"""
Main driver program.

Author(s):
    Michael Yao

Licensed under the MIT License. Copyright University of Pennsylvania 2023.
"""
import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import wandb

from data.mnist import MNISTDataModule
from models.generator import GeneratorModule
from experiment.params import Experiment
from experiment.utility import seed_everything


def main():
    exp = Experiment()
    seed_everything(seed=exp.seed)
    beta1, beta2 = exp.beta

    datamodule = MNISTDataModule(
        batch_size=exp.batch_size, num_workers=exp.num_workers, seed=exp.seed
    )
    model = GeneratorModule(
        objective=exp.objective,
        alpha=exp.alpha,
        lr=exp.lr,
        beta1=beta1,
        beta2=beta2
    )

    callbacks = [
        EarlyStopping(
            monitor="val_objective_G",
            min_delta=0.0,
            patience=10,
            verbose=False,
            mode="max"
        ),
        ModelCheckpoint(
            dirpath=exp.ckpt_dir,
            monitor="val_objective_G",
            filename="{epoch}_{val_objective_G:.2f}_{val_loss_D:.2f}",
            save_last=True,
            mode="max"
        )
    ]

    logger = False
    if not exp.disable_wandb and not exp.fast_dev_run:
        logger = WandbLogger(
            project=os.path.basename(os.path.dirname(__file__)),
            log_model="all",
        )
        logger.log_hyperparams(exp.args)
        logger.watch(model)

    trainer = pl.Trainer(
        max_epochs=exp.num_epochs,
        accelerator=exp.device,
        logger=logger,
        callbacks=callbacks,
        deterministic=True,
        fast_dev_run=exp.fast_dev_run,
        gradient_clip_val=exp.clip,
    )

    if exp.mode in ("both", "train"):
        trainer.fit(
            model,
            datamodule=datamodule,
            ckpt_path=exp.resume_from
        )
    if exp.mode in ("both", "test"):
        model = GeneratorModule.load_from_checkpoint(exp.resume_from)
        trainer.test(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
