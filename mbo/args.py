"""
Defines the experimental arguments for offline MBO baseline method evaluation.

Author(s):
    Michael Yao @michael-s-yao

Licensed under the MIT License. Copyright University of Pennsylvania 2024.
"""
import argparse
import design_bench


def build_args() -> argparse.Namespace:
    """
    Defines the experimental arguments for offline MBO baseline method
    evaluation.
    Input:
        None.
    Returns:
        A namespace containing the experimental argument values.
    """
    parser = argparse.ArgumentParser(
        description="Offline MBO Baseline Method Evaluation"
    )

    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=[task.task_name for task in design_bench.registry.all()],
        help="The name of the design-bench task for the experiment."
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="Optional constant value of alpha. Default our method."
    )
    parser.add_argument(
        "--logging-dir",
        type=str,
        default=None,
        help="Logging directory to save optimization results to."
    )
    parser.add_argument(
        "--device", type=str, default="cpu", help="Device. Default CPU."
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed. Default 42."
    )
    parser.add_argument(
        "--smart-start",
        action="store_true",
        help="Whether to initilize the GP with the best points in the dataset."
    )
    parser.add_argument(
        "--ablate-source-critic-training",
        action="store_true",
        help="Whether to keep the source critic fixed during optimization."
    )

    return parser.parse_args()
