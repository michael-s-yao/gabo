"""
Main driver program for warfarin conterfactual generation training.

Author(s):
    Michael Yao

Licensed under the MIT License. Copyright University of Pennsylvania 2023.
"""
import torch
import lightning.pytorch as pl
from lightning.pytorch.strategies.ddp import DDPStrategy
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger


from data.iwpc import IWPCWarfarinDataModule
from experiment.warfarin_params import Experiment
from models.ctgan import CTGANLightningModule
from experiment.utility import seed_everything


def main():
    exp = Experiment()
    seed_everything(exp.seed)
    datamodule = IWPCWarfarinDataModule(
        root=exp.datadir,
        cv_idx=exp.cv_idx,
        batch_size=exp.batch_size,
        num_workers=exp.num_workers,
        label_smoothing=exp.label_smoothing,
        seed=exp.seed
    )
    datamodule.prepare_data()
    datamodule.setup()

    dummy = next(iter(datamodule.train_dataloader()))
    patient_vector_dim = dummy.X.size(dim=-1)
    condition_mask_dim = dummy.cond_mask.size(dim=-1)
    model = CTGANLightningModule(
        patient_vector_dim=patient_vector_dim,
        condition_mask_dim=condition_mask_dim,
        invert_continuous_transform=datamodule.invert,
        alpha=exp.alpha,
        embedding_dim=64,
        generator_dims=[256, 256],
        critic_dims=[512, 512],
        optimizer=exp.optimizer,
        lr=exp.lr,
        weight_decay=exp.weight_decay,
        batch_size=exp.batch_size
    )

    meta = f"warfarin_alpha={exp.alpha}"
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
            project="Warfarin",
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
        # callbacks=callbacks,
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
        model = CTGANLightningModule.load_from_checkpoint(
            exp.resume_from, map_location=torch.device("cpu")
        )
        trainer.test(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
