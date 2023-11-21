"""
Main driver program for warfarin counterfactual generation.

Author(s):
    Michael Yao @michael-s-yao

Licensed under the MIT License. Copyright University of Pennsylvania 2023.
"""
import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import sys
import torch
import torch.optim as optim
from tqdm import tqdm

sys.path.append(".")
from warfarin.cost import dosage_cost
from warfarin.dataset import WarfarinDataset
from warfarin.dosing import WarfarinDose
from warfarin.policy import DosingPolicy
from warfarin.transform import PowerNormalizeTransform
from warfarin.metrics import Divergence, SupportCoverage
from models.lipschitz import FrozenMLPRegressor
from models.critic import WeightClipper
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
        "--max_critic_per_epoch",
        type=int,
        default=500,
        help="Maximum number of iterative steps to train the critic per epoch."
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
        "--patience",
        type=int,
        default=-1,
        help="Patience for early stopping. By default, no early stopping."
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
        "--num_in_distribution",
        type=int,
        default=16,
        help="Number of in-distribution samples used to estimate OOD distance."
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
    rng = np.random.RandomState(args.seed)

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

    best_cost, best_epoch = 1e12, None
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
            pred_costs = transform.invert(np.squeeze(surrogate(X_test)))
            cost = np.mean(pred_costs)
            preds.append((cost, np.std(pred_costs, ddof=1)))

            gt_costs = dosage_cost(
                col_transforms[dataset.dose].invert(X_test[dataset.dose]),
                oracle(X_test)
            )
            gts.append((np.mean(gt_costs), np.std(gt_costs, ddof=1)))

            if args.patience > 0:
                if cost < best_cost:
                    best_cost, best_epoch = cost, i
                if i - best_epoch > args.patience:
                    break

            # Fit the source critic.
            if i % 4 == 0:
                policy.fit_critic(X_test)

    print("Surrogate Cost:", preds[-1][0], preds[-1][1] / np.sqrt(len(X_test)))
    print("Oracle Cost:", gts[-1][0], gts[-1][1] / np.sqrt(len(X_test)))
    doses_gen = col_transforms[dataset.dose].invert(policy.optimum())
    doses_true = col_transforms[dataset.dose].invert(X_test[dataset.dose])
    print("Support Coverage:", SupportCoverage(doses_true, doses_gen))
    print("JS Divergence:", Divergence(doses_true, doses_gen))

    # Plot relevant metrics.
    preds = (
        np.array([mean for mean, _ in preds]),
        np.array([std for _, std in preds]) / np.sqrt(len(X_test))
    )
    gts = (
        np.array([mean for mean, _ in gts]),
        np.array([std for _, std in gts]) / np.sqrt(len(X_test))
    )
    plt.figure(figsize=(10, 5))
    steps = args.batch_size * np.arange(len(preds[0]))
    for (mean, sem), label in zip([preds, gts], ["Surrogate", "Oracle"]):
        plt.plot(steps, mean, label=label)
        plt.fill_between(steps, mean - sem, mean + sem, alpha=0.1)
    plt.xlabel("Optimization Steps")
    plt.ylabel("Warfarin-Associated Dosage Cost")
    plt.xlim(np.min(steps), np.max(steps))
    plt.ylim(0, 1000)
    plt.legend()
    if args.savepath is None:
        plt.show()
    else:
        plt.savefig(
            args.savepath, dpi=600, transparent=True, bbox_inches="tight"
        )
    plt.close()
    return


if __name__ == "__main__":
    seed_everything()
    torch.set_default_dtype(torch.float64)
    main()
