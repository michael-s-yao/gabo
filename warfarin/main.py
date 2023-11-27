"""
Main driver program for warfarin counterfactual generation.

Author(s):
    Michael Yao @michael-s-yao

Licensed under the MIT License. Copyright University of Pennsylvania 2023.
"""
import argparse
import json
import numpy as np
import pickle
import sys
import torch
import warnings
from math import ceil
from botorch.utils.transforms import unnormalize
from botorch.exceptions.warnings import (
    BadInitialCandidatesWarning, InputDataWarning
)

sys.path.append(".")
from warfarin.cost import dosage_cost
from warfarin.dataset import WarfarinDataset
from warfarin.dosing import WarfarinDose
from warfarin.policy import DosingPolicy
from warfarin.transform import PowerNormalizeTransform
from models.lipschitz import FrozenMLPRegressor
from experiment.utility import seed_everything


def build_args() -> argparse.Namespace:
    """
    Builds arguments for warfarin counterfactual generation experiments.
    Input:
        None.
    Returns:
        Namespace of arguments for experiments.
    """
    parser = argparse.ArgumentParser(description="Warfarin Dosage Policy")

    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="Relative source critic regularization weighting."
    )
    parser.add_argument(
        "--hparams",
        type=str,
        default="./warfarin/hparams.json",
        help="Path to JSON file with source critic hyperparameters."
    )
    parser.add_argument(
        "--savepath",
        type=str,
        default=None,
        help="Path to save the optimization results to. Default not saved."
    )
    parser.add_argument(
        "--thresh_max",
        type=float,
        default=315,
        help="Maximum safe warfarin dose in units of mg/week. Default 315."
    )
    parser.add_argument(
        "--batch_size", type=int, default=8, help="Batch size. Default 8."
    )
    ngpc_help = "Number of times to sample doses before retraining the source "
    ngpc_help += "critic. Default 2."
    parser.add_argument(
        "--num_generator_per_critic", type=int, default=2, help=ngpc_help
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=256,
        help="Total sampling budget per patient. Default 256."
    )
    parser.add_argument(
        "--device", type=str, default="cpu", help="Device. Default CPU."
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed. Default 42."
    )

    return parser.parse_args()


def main():
    args = build_args()
    torch.set_default_dtype(torch.float64)
    torch.set_default_device(args.device)
    device = torch.device(args.device)
    use_deterministic = "cuda" not in args.device
    seed_everything(seed=args.seed, use_deterministic=use_deterministic)

    warnings.filterwarnings("ignore", category=BadInitialCandidatesWarning)
    warnings.filterwarnings("ignore", category=InputDataWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    # Load the surrogate objective function.
    dataset = WarfarinDataset(seed=args.seed)
    oracle = WarfarinDose()
    surrogate = FrozenMLPRegressor(
        f"./warfarin/docs/MLPRegressor_cost_{args.seed}.pkl"
    )
    with open(args.hparams, "rb") as f:
        hparams = json.load(f)
        critic_hparams = hparams["SourceCritic"]

    # Initialize the sampling policy and data transforms.
    X_train, pred_dose_train = dataset.train_dataset
    X_test, pred_dose_test = dataset.test_dataset
    col_transforms = {}
    for col in [dataset.height, dataset.weight, dataset.dose]:
        t_col = PowerNormalizeTransform(X_train, p=1, key=col)
        X_train = t_col(X_train)
        X_test = t_col(X_test)
        col_transforms[col] = t_col
    z_range = col_transforms[dataset.dose](np.array([0, args.thresh_max]))

    policy = DosingPolicy(
        ref_dataset=X_train.astype(np.float64),
        alpha=args.alpha,
        surrogate=surrogate,
        min_z_dose=np.min(z_range),
        max_z_dose=np.max(z_range),
        seed=args.seed,
        batch_size=args.batch_size,
        device=device,
        **critic_hparams
    )

    # Choose the initial set of observations.
    a = []
    z = unnormalize(
        torch.rand(X_test.shape[0], args.batch_size, device=policy.device),
        bounds=torch.tensor(
            [[policy.min_z_dose], [policy.max_z_dose]], device=policy.device
        )
    )
    # policy.fit_critic(X_test, z)
    alpha = policy.alpha(X_test)
    a.append(alpha)
    y = (1.0 - alpha) * policy.surrogate_cost(X_test, z) + (
        alpha * policy.wasserstein(X_test, z)
    )
    y_gt = dosage_cost(
        col_transforms[dataset.dose].invert(
            z.flatten().detach().cpu().numpy()
        ),
        np.tile(oracle(X_test), y.size(dim=-1))
    )
    y_gt = y_gt.reshape(*tuple(y.size()))

    # Generative adversarial Bayesian optimization.
    for step in range(ceil(args.budget / args.batch_size) - 1):
        policy.fit(z.detach(), y.detach())

        # Optimize and get new observations.
        new_z = policy(X_test, y, step)

        # Calculate the surrogate and oracle objective values.
        alpha = policy.alpha(X_test)
        a.append(alpha)
        new_y = (1.0 - alpha) * policy.surrogate_cost(X_test, new_z) + (
            alpha * policy.wasserstein(X_test, new_z)
        )
        new_y_gt = dosage_cost(
            col_transforms[dataset.dose].invert(
                new_z.flatten().detach().cpu().numpy()
            ),
            np.tile(oracle(X_test), new_y.size(dim=-1))
        )
        new_y_gt = new_y_gt.reshape(*tuple(new_y.size()))

        # Update training points.
        z = torch.cat((z, new_z), dim=-1)
        y = torch.cat((y, new_y), dim=-1)
        y_gt = np.concatenate((y_gt, new_y_gt), axis=-1)

        # Fit the source critic.
        if step % args.num_generator_per_critic == 0:
            policy.fit_critic(X_test, z)

    # Save optimization results.
    if args.savepath is not None:
        with open(args.savepath, mode="wb") as f:
            results = {
                "z": z.detach().cpu().numpy(),
                "y": y.detach().cpu().numpy(),
                "y_gt": y_gt,
                "alpha": np.array(a),
                "X": X_test,
                "batch_size": args.batch_size
            }
            pickle.dump(results, f)


if __name__ == "__main__":
    main()
