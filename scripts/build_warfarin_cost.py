"""
Trains and tests a mortality risk estimator based on patient variables and INR
for patients taking warfarin.

Author(s):
    Michael Yao

Licensed under the MIT License. Copyright University of Pennsylvania 2023.
"""
import sys
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import kstest
from pathlib import Path
import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from typing import Optional, Union

sys.path.append(".")
from data.iwpc import IWPCWarfarinDataModule
from models.mortality_estimator import WarfarinMortalityLightningModule
from experiment.warfarin_params import build_warfarin_mortality_estimator_args
from experiment.utility import seed_everything, plot_config


def plot_test_results(
    ckpt: Union[Path, str],
    savepath: Optional[Union[Path, str]] = None,
    seed: int = 42
) -> None:
    plot_config(fontsize=18)
    device = torch.device("cpu")

    datamodule = IWPCWarfarinDataModule(
        root="./data/warfarin",
        train_test_split=(0.8, 0.2),
        cv_idx=-1,
        seed=seed,
        training_by_sampling=False
    )
    datamodule.prepare_data()
    datamodule.setup()

    model = WarfarinMortalityLightningModule.load_from_checkpoint(
        ckpt, map_device=device
    )
    model = model.to(device)

    plt.figure(figsize=(10, 6))
    train, test = None, None
    for dataset, label in zip(
        [datamodule.train, datamodule.test], ["Train Dataset", "Test Dataset"]
    ):
        gts = np.array([pt.cost for pt in dataset])
        preds = np.array([model(pt.X.to(device)).item() for pt in dataset])
        vals = (preds - gts) / ((preds + gts) / 2.0)
        if label == "Train Dataset":
            train = vals
        else:
            test = vals
        plt.hist(
            vals,
            label=(label + rf" ($N = {len(dataset)}$)"),
            density=True,
            alpha=0.5,
            bins=np.linspace(-1, 1, 100),
            ec="black"
        )
    plt.legend()
    plt.xlabel("True Cost - Predicted Cost")
    plt.ylabel("Density")
    if savepath is None:
        plt.show()
    else:
        plt.savefig(savepath, dpi=600, bbox_inches="tight", transparent=True)
    plt.close()
    p, N = 0.0, 100
    for _ in range(N):
        p += kstest(test, train, method='exact').pvalue / N
    print(f"p = {p:.5f} (Kolmogorov-Smirnov Test)")


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
        seed=args.seed,
        training_by_sampling=False
    )
    datamodule.prepare_data()
    datamodule.setup()

    dummy = datamodule.train[0]
    dummy = datamodule.invert(
        torch.unsqueeze(dummy.X, dim=0), dummy.X_attributes
    )
    model = WarfarinMortalityLightningModule(
        in_dim=dummy.size(dim=-1),
        hidden_dims=args.hidden_dims,
        invert_continuous_transform=datamodule.invert,
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
    if (
        not args.fast_dev_run and
        not args.mode == "test" and
        not args.disable_wandb
    ):
        logger = WandbLogger(project="WarfarinCostEstimator", log_model="all")
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
    reproduce_warfarin_cost_estimator_graph = False
    if reproduce_warfarin_cost_estimator_graph:
        plot_test_results(
            ckpt="./ckpts/warfarin_cost_estimator.ckpt",
            savepath="./docs/warfarin_cost_estimator.png"
        )
    main()
