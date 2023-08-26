"""
Experimental parameter class.

Author(s):
    Michael Yao

Licensed under the MIT License. Copyright University of Pennsylvania 2023.
"""
import argparse


class Experiment:
    """Experimental parameter class."""

    def __init__(self):
        """
        Args:
            None.
        """
        parser = argparse.ArgumentParser(
            description="OOD Optimization Experiments"
        )

        objective_help = "Objective function to optimize against. "
        objective_help += "One of [`gradient`, `energy`]."
        parser.add_argument(
            "--objective",
            type=str,
            required=True,
            choices=("gradient", "grad", "energy"),
            help=objective_help
        )
        parser.add_argument(
            "--alpha",
            default=0.0,
            type=float,
            help="Source discriminator regularization weighting. Default 0.0."
        )
        parser.add_argument(
            "--resume_from",
            "--ckpt",
            default=None,
            type=str,
            help="Checkpoint file to resume training from. Default None."
        )
        parser.add_argument(
            "--lr",
            default=0.0002,
            type=float,
            help="Learning rate. Default 0.0002."
        )
        beta_help = "Beta parameters for Adam optimizer. "
        beta_help += "Default beta_1 = 0.5, beta_2 = 0.999."
        parser.add_argument(
            "--beta", default=[0.5, 0.999], type=float, nargs=2, help=beta_help
        )
        parser.add_argument(
            "--clip",
            default=None,
            type=float,
            help="Gradient clipping. Default no clipping."
        )
        parser.add_argument(
            "--num_epochs",
            default=100,
            type=int,
            help="Number of epochs. Default 100."
        )
        parser.add_argument(
            "--ckpt_dir",
            default="./ckpts",
            type=str,
            help="Directory to save model checkpoint. Default ./ckpts."
        )
        parser.add_argument(
            "--batch_size",
            default=128,
            type=int,
            help="Batch size. Default 128."
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
            "--mode",
            default="train",
            type=str,
            choices=("train", "test", "both"),
            help="Whether to train the model, test a model, or both."
        )
        parser.add_argument(
            "--fast_dev_run", action="store_true", help="Test code only."
        )
        parser.add_argument(
            "--disable_wandb",
            action="store_true",
            help="Disable Weights and Biases logging."
        )

        self.args = parser.parse_args()

        _ = [
            setattr(self, key, val) for key, val in self.args.__dict__.items()
        ]
