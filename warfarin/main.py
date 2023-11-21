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
from tqdm import tqdm

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
        "--surrogate_cost",
        type=str,
        default="./warfarin/docs/MLPRegressor_cost.pkl",
        help="Path to surrogate cost function."
    )
    parser.add_argument(
        "--hparams",
        type=str,
        default="./warfarin/hparams.json",
        help="Path to JSON file with surrogate cost hyperparameters."
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=100,
        help="Number of batches to sample and optimizer over. Default 100."
    )
    parser.add_argument(
        "--savepath",
        type=str,
        default=None,
        help="Path to save the optimization results to. Default not saved."
    )
    parser.add_argument(
        "--min_z_dose",
        type=float,
        default=-10.0,
        help="Minimum warfarin dose to search over in normalized units."
    )
    parser.add_argument(
        "--max_z_dose",
        type=float,
        default=10.0,
        help="Maximum warfarin dose to search over in normalized units."
    )
    parser.add_argument(
        "--thresh_max",
        type=float,
        default=315,
        help="Maximum safe warfarin dose in units of mg/week. Default 315."
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Batch size. Default 16."
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed. Default 42."
    )

    return parser.parse_args()


def main():
    args = build_args()

    # Load the surrogate objective function.
    dataset = WarfarinDataset(seed=args.seed)
    oracle = WarfarinDose()
    surrogate = FrozenMLPRegressor(args.surrogate_cost)
    with open(args.hparams, "rb") as f:
        hparams = json.load(f)
        critic_hparams = hparams["SourceCritic"]
        p = hparams["Surrogate"]["p"]

    # Initialize the sampling policy and data transforms.
    X_train, pred_dose_train = dataset.train_dataset
    X_test, pred_dose_test = dataset.test_dataset
    gt_dose_train = oracle(X_train)
    cost_train = dosage_cost(pred_dose_train, gt_dose_train)
    transform = PowerNormalizeTransform(cost_train, p)

    col_transforms = {}
    for col in [dataset.height, dataset.weight, dataset.dose]:
        t_col = PowerNormalizeTransform(X_train, p=1, key=col)
        X_train = t_col(X_train)
        X_test = t_col(X_test)
        col_transforms[col] = t_col

    # Choose the initial set of observations.
    z_range = col_transforms[dataset.dose](np.array([0, args.thresh_max]))
    min_z_dose = max(np.min(z_range), args.min_z_dose)
    max_z_dose = min(np.max(z_range), args.max_z_dose)
    a = []
    policy = DosingPolicy(
        ref_dataset=X_train.astype(np.float64),
        surrogate=surrogate,
        min_z_dose=min_z_dose,
        max_z_dose=max_z_dose,
        seed=args.seed,
        **critic_hparams
    )
    X_test = policy(X_test)
    policy.fit_critic(X_test)

    preds, gts = [], []
    with tqdm(
        range(args.num_epochs), desc="Optimizing Warfarin Dose", leave=False
    ) as pbar:
        for i, _ in enumerate(pbar):
            # Sample according to the policy.
            for _ in range(args.batch_size):
                X_test = policy(X_test)
                alpha = policy.alpha()
                cost = surrogate(
                    torch.from_numpy(X_test.to_numpy().astype(np.float64))
                )
                penalty = torch.unsqueeze(
                    policy.wasserstein(
                        torch.from_numpy(
                            policy.dataset.to_numpy().astype(np.float64)
                        ),
                        torch.from_numpy(
                            X_test.to_numpy().astype(np.float64)
                        )
                    ),
                    dim=-1
                )
                y = ((1.0 - alpha) * cost) + (alpha * penalty)
                y = torch.squeeze(y, dim=-1).detach().cpu().numpy()
                policy.feedback(y)

            optimal_doses = policy.optimum()
            X_test[policy.dose_key] = optimal_doses
            policy.reset()

            # Calculate statistics.
            preds.append(transform.invert(np.squeeze(surrogate(X_test))))

            gt_costs = dosage_cost(
                col_transforms[dataset.dose].invert(X_test[dataset.dose]),
                oracle(X_test)
            )
            gts.append(gt_costs)

            # Fit the source critic.
            if i % 4 == 0:
                policy.fit_critic(X_test)

    # Save optimization results.
    if args.savepath is not None:
        with open(args.savepath, mode="wb") as f:
            results = {
                "X": X_test,
                "preds": preds,
                "gt": gts,
                "alpha": np.array(a),
                "batch_size": args.batch_size,
            }
            pickle.dump(results, f)


if __name__ == "__main__":
    seed_everything()
    torch.set_default_dtype(torch.float64)
    main()
