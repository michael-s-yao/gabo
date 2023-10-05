"""
Trains and tests a mortality risk estimator based on patient variables and INR
for patients taking warfarin.

Author(s):
    Michael Yao

Licensed under the MIT License. Copyright University of Pennsylvania 2023.
"""
import sys
import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

sys.path.append(".")
from data.iwpc import IWPCWarfarinDataModule
from models.mortality_estimator import WarfarinMortalityLightningModule
from experiment.warfarin_params import build_warfarin_mortality_estimator_args
from experiment.utility import seed_everything


def main():
    args = build_warfarin_mortality_estimator_args()
    seed_everything(seed=args.seed)

    datamodule = IWPCWarfarinDataModule(
        root=args.datadir,
        train_test_split=(0.8, 0.2),
        cv_idx=args.cv_idx,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        label_smoothing=args.label_smoothing,
        seed=args.seed
    )
    datamodule.prepare_data()
    datamodule.setup()

    model = WarfarinMortalityLightningModule(
        in_dim=datamodule.train[0].X.size(dim=0),
        hidden_dims=args.hidden_dims,
        dropout=args.dropout,
        lr=args.lr,
        lr_milestones=args.lr_milestones,
        lr_gamma=args.lr_gamma,
        optimizer=args.optimizer,
        beta=args.beta,
        weight_decay=args.weight_decay
    )

    callbacks = [
        ModelCheckpoint(
            dirpath=args.ckpt_dir,
            monitor="val_loss",
            mode="min",
            save_last=True
        ),
        EarlyStopping(
            monitor="val_loss",
            min_delta=0.0,
            patience=10,
            verbose=False,
            mode="min"
        )
    ]

    logger = False
    if not args.fast_dev_run and not args.mode == "test":
        logger = CSVLogger(
            save_dir="./",
        )
        logger.log_hyperparams(args)

    trainer = pl.Trainer(
        max_epochs=args.num_epochs,
        accelerator=args.device,
        logger=logger,
        callbacks=callbacks,
        deterministic=True,
        fast_dev_run=args.fast_dev_run,
        devices=args.devices
    )

    if args.mode in ("both", "train"):
        trainer.fit(
            model,
            datamodule=datamodule,
            ckpt_path=args.resume_from
        )
    if args.mode in ("both", "test"):
        model = WarfarinMortalityLightningModule.load_from_checkpoint(
            args.resume_from, map_location=torch.device("cpu")
        )
        trainer.test(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
