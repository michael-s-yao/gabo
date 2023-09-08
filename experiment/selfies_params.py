"""
Experimental parameter class for SELFIES molecule generation task.

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
            description="Molecule Generation OOD Optimization Experiments"
        )

        parser.add_argument(
            "--objective",
            type=str,
            required=True,
            help="Objective function to optimize against."
        )
        parser.add_argument(
            "--alpha",
            default=0.5,
            type=float,
            help="Relative regularization weighting. Default 0.5."
        )
        parser.add_argument(
            "--regularization",
            default="gan_loss",
            type=str,
            choices=(
                "gan_loss",
                "importance_weighting",
                "log_importance_weighting",
                "wasserstein",
                "em"
            ),
            help="In-distribution regularization. Default `gan_loss`."
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
        beta_help += "Default beta_1 = 0.9, beta_2 = 0.999."
        parser.add_argument(
            "--beta", default=[0.9, 0.999], type=float, nargs=2, help=beta_help
        )
        parser.add_argument(
            "--clip",
            default=None,
            type=float,
            help="Gradient clipping. Default no clipping."
        )
        parser.add_argument(
            "--n_critic_per_generator",
            default=1.0,
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
            default="./ckpts",
            type=str,
            help="Directory to save model checkpoint. Default ./ckpts."
        )
        parser.add_argument(
            "--batch_size",
            default=64,
            type=int,
            help="Batch size. Default 64."
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

        self.args = parser.parse_args()

        _ = [
            setattr(self, key, val) for key, val in self.args.__dict__.items()
        ]
