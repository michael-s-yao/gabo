"""
Experimental parameter class for warfarin dosage estimation task.

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
            description="Warfarin Counterfactual OOD Optimization Experiments"
        )

        parser.add_argument(
            "--alpha",
            default=0.5,
            type=float,
            help="Relative regularization weighting. Default 0.5."
        )
        parser.add_argument(
            "--resume_from",
            "--ckpt",
            default=None,
            type=str,
            help="Checkpoint file to resume training from. Default None."
        )
        parser.add_argument(
            "--datadir",
            default="./data/warfarin",
            type=str,
            help="Data directory. Default `./data/warfarin`."
        )
        parser.add_argument(
            "--cv_idx",
            default=0,
            type=int,
            help="Cross validation index. Default 0."
        )
        parser.add_argument(
            "--lr",
            default=0.002,
            type=float,
            help="Learning rate. Default 0.002."
        )
        parser.add_argument(
            "--optimizer",
            default="Adam",
            type=str,
            choices=("SGD", "Adam", "RMSProp"),
            help="Optimizer algorithm. Default Adam optimizer."
        )
        parser.add_argument(
            "--beta",
            default=[0.9, 0.999],
            type=float,
            nargs="+",
            help="Betas/momentum optimizer hyperparameter."
        )
        parser.add_argument(
            "--weight_decay",
            default=0.001,
            type=float,
            help="Weight decay. Default 0.001."
        )
        parser.add_argument(
            "--label_smoothing",
            default=0.01,
            type=float,
            help="Label smoothing. Default 0.01."
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
            "--devices",
            default=1,
            type=int,
            nargs="+",
            help="Devices. Default auto."
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


def build_warfarin_mortality_estimator_args() -> argparse.Namespace:
    """
    Builds the argument Namespace for training and evaluating a mortality cost
    estimator based on INR for patients on warfarin.
    Input:
        None.
    Returns:
        The relevant Namespace with the specified arguments.
    """
    parser = argparse.ArgumentParser(
        description="Mortality cost estimator for patients on warfarin."
    )

    parser.add_argument(
        "--hidden_dims",
        default=[32],
        type=int,
        nargs="+",
        help="Hidden dimensions for intermediate layers of the FCNN model."
    )
    parser.add_argument(
        "--datadir",
        default="./data/warfarin",
        type=str,
        help="Data directory. Default `./data/warfarin`."
    )
    cv_idx_help = "Cross validation index. Default 0. "
    cv_idx_help += "If -1, train and val sets will be combined for training."
    parser.add_argument(
        "--cv_idx", default=0, type=int, help=cv_idx_help
    )
    parser.add_argument(
        "--lr",
        default=0.001,
        type=float,
        help="Learning rate. Default 0.001."
    )
    parser.add_argument(
        "--lr_milestones",
        default=[20, 40],
        type=int,
        nargs="+",
        help="Learning rate milestones for LR scheduler. Default [20, 40]."
    )
    parser.add_argument(
        "--lr_gamma",
        default=0.5,
        type=float,
        help="Learning rate decay rate at specified milestones. Default 0.5."
    )
    parser.add_argument(
        "--optimizer",
        default="Adam",
        type=str,
        choices=("SGD", "Adam", "RMSProp"),
        help="Optimizer algorithm. Default Adam optimizer."
    )
    parser.add_argument(
        "--beta",
        default=[0.9, 0.999],
        type=float,
        nargs="+",
        help="Betas/momentum optimizer hyperparameter."
    )
    parser.add_argument(
        "--dropout",
        default=0.1,
        type=float,
        help="Dropout parameter. Default 0.1."
    )
    parser.add_argument(
        "--weight_decay",
        default=0.001,
        type=float,
        help="Weight decay. Default 0.001."
    )
    parser.add_argument(
        "--label_smoothing",
        default=0.01,
        type=float,
        help="Label smoothing. Default 0.01."
    )
    parser.add_argument(
        "--num_epochs",
        default=50,
        type=int,
        help="Number of epochs. Default 50."
    )
    parser.add_argument(
        "--ckpt_dir",
        default="./ckpts",
        type=str,
        help="Directory to save model checkpoint. Default ./ckpts."
    )
    parser.add_argument(
        "--resume_from",
        "--ckpt",
        default=None,
        type=str,
        help="Checkpoint file to resume training from. Default None."
    )
    parser.add_argument(
        "--batch_size",
        default=16,
        type=int,
        help="Batch size. Default 16."
    )
    parser.add_argument(
        "--seed", default=42, type=int, help="Random seed. Default 42."
    )
    parser.add_argument(
        "--device", default="auto", type=str, help="Device. Default auto."
    )
    parser.add_argument(
        "--devices",
        default=1,
        type=int,
        nargs="+",
        help="Devices. Default auto."
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

    return parser.parse_args()
