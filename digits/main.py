"""
Main driver program for MNIST-related training.

Author(s):
    Michael Yao

Licensed under the MIT License. Copyright University of Pennsylvania 2023.
"""
import argparse
import os
import sys
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import WandbLogger

sys.path.append(".")
from digits.data.mnist import MNISTDataModule
from digits.wgan import WGANModule
from experiment.utility import seed_everything


def build_args() -> argparse.Namespace:
    """
    Builds the relevant parameters for the MNIST digit image generation task.
    Input:
        None.
    Returns:
        A namespace with the relevant parameters for the MNIST experiments.
    """
    parser = argparse.ArgumentParser(
        description="MNIST Generative Adversarial Optimization Experiments"
    )

    alpha_help = "Relative regularization weighting. Use `Lipschitz` for our "
    alpha_help += "method, otherwise specify a float between 0 and 1."
    parser.add_argument("--alpha", type=str, required=True, help=alpha_help)
    parser.add_argument(
        "--weight_decay", default=1e-6, type=float, help="Weight decay."
    )
    parser.add_argument(
        "--lr",
        default=0.0001,
        type=float,
        help="Learning rate. Default 0.0001."
    )
    parser.add_argument(
        "--clip",
        default=None,
        type=float,
        help="Gradient clipping. Default no clipping."
    )
    parser.add_argument(
        "--n_critic_per_generator",
        default=5,
        type=float,
        help="Number of times to optimize the critic versus the generator."
    )
    parser.add_argument(
        "--num_epochs",
        default=200,
        type=int,
        help="Number of epochs. Default 200."
    )
    parser.add_argument(
        "--ckpt_dir",
        default="./digits/ckpts",
        type=str,
        help="Directory to save model checkpoint. Default ./digits/ckpts"
    )
    parser.add_argument(
        "--batch_size",
        default=256,
        type=int,
        help="Batch size. Default 256."
    )
    parser.add_argument(
        "--seed", default=42, type=int, help="Random seed. Default 42."
    )
    parser.add_argument(
        "--device", default="auto", type=str, help="Device. Default auto."
    )
    parser.add_argument(
        "--num_workers",
        default=0,
        type=int,
        help="Number of workers. Default 0."
    )
    parser.add_argument(
        "--find_unused_parameters",
        action="store_true",
        help="Find unused parameters in distributed multi-GPU training."
    )
    parser.add_argument(
        "--fast_dev_run", action="store_true", help="Test code only."
    )
    parser.add_argument(
        "--disable_wandb",
        action="store_true",
        help="Disable Weights and Biases logging."
    )

    return parser.parse_args()


def main():
    args = build_args()
    seed_everything(seed=args.seed)

    dm = MNISTDataModule(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed
    )
    model = WGANModule(
        alpha=args.alpha,
        lr=args.lr,
        weight_decay=args.weight_decay,
        clip=args.clip
    )

    callbacks = [
        ModelCheckpoint(
            dirpath=args.ckpt_dir, save_last=True, monitor="val_loss_G"
        )
    ]

    logger = False
    if not args.fast_dev_run and not args.disable_wandb:
        logger = WandbLogger(
            project=os.path.basename(os.path.dirname(__file__)),
            log_model=True,
        )
        logger.log_hyperparams(args)
    strategy = "auto"
    if args.find_unused_parameters:
        strategy = DDPStrategy(find_unused_parameters=True)
    trainer = pl.Trainer(
        max_epochs=args.num_epochs,
        accelerator=args.device,
        callbacks=callbacks,
        logger=logger,
        deterministic=True,
        fast_dev_run=args.fast_dev_run,
        strategy=strategy,
        devices=[0]
    )

    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    main()
