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
import pickle
from tqdm import tqdm
from sklearn.linear_model import Lasso
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from cost import dosage_cost
from dataset import WarfarinDataset
from dosing import WarfarinDose
from sampler import RandomSampler


def build_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Warfarin Dosage Policy")

    parser.add_argument(
        "--surrogate_cost",
        type=str,
        default="./docs/MLPRegressor_cost.pkl",
        help="Path to surrogate cost function."
    )
    parser.add_argument(
        "--hparams",
        type=str,
        default="./hparams.json",
        help="Path to JSON file with surrogate cost hyperparameters."
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=500,
        help="Number of batches to sample and optimizer over. Default 500."
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
        "--batch_size", type=int, default=16, help="Batch size. Default 16."
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed. Default 42."
    )

    return parser.parse_args()


def main():
    args = build_args()

    dataset = WarfarinDataset(seed=args.seed)
    dose = WarfarinDose()
    with open(args.surrogate_cost, "rb") as f:
        cost_func = pickle.load(f)
    with open(args.hparams, "rb") as f:
        p = json.load(f)[str(type(cost_func)).split(".")[-1][:-2]]["p"]

    X_train, pred_dose_train = dataset.train_dataset
    gt_dose_train = dose(X_train)
    X_test, pred_dose_test = dataset.test_dataset
    gt_dose_test = dose(X_test)
    cost_train = dosage_cost(pred_dose_train, gt_dose_train)
    cost_train_p = np.power(cost_train, p)
    scaler_cost = StandardScaler()
    cost_train = scaler_cost.fit_transform(cost_train_p.to_numpy()[:, None])
    cost_test = dosage_cost(pred_dose_test, gt_dose_test)

    if isinstance(cost_func, (Lasso, MLPRegressor)):
        scaler_h, scaler_w = StandardScaler(), StandardScaler()
        scaler_d = StandardScaler()
        for col, scaler in zip(
            [dataset.height, dataset.weight, dataset.dose],
            [scaler_h, scaler_w, scaler_d]
        ):
            X_train[col] = scaler.fit_transform(
                X_train[col].to_numpy()[:, None]
            )
            X_test[col] = scaler.transform(X_test[col].to_numpy()[:, None])

    policy = RandomSampler(min_z_dose=-2.0, max_z_dose=2.0, seed=args.seed)
    X = X_train
    best_cost, best_epoch = 1e12, None
    preds, gts = [], []
    with tqdm(
        range(args.num_epochs), desc="Optimizing Warfarin Dose", leave=False
    ) as pbar:
        for i, _ in enumerate(pbar):
            pred_costs = np.squeeze(cost_func.predict(X))[:, None]
            pred_costs = np.power(
                np.squeeze(scaler_cost.inverse_transform(pred_costs)), 1.0 / p
            )
            _ = [
                policy.feedback(cost_func.predict(policy(X)))
                for _ in range(args.batch_size)
            ]
            optimal_doses = policy.optimum()
            X[policy.dose_key] = optimal_doses
            policy.reset()
            cost = np.mean(pred_costs)
            preds.append((cost, np.std(pred_costs, ddof=1)))
            gt_costs = dosage_cost(X[policy.dose_key], dose(X))
            gts.append((np.mean(gt_costs), np.std(gt_costs, ddof=1)))
            pbar.set_postfix(cost=cost)
            if args.patience > 0:
                if cost < best_cost:
                    best_cost, best_epoch = cost, i
                if i - best_epoch > args.patience:
                    break

    preds = (
        np.array([mean for mean, _ in preds]),
        np.array([std for _, std in preds]) / np.sqrt(len(preds))
    )
    gts = (
        np.array([mean for mean, _ in gts]),
        np.array([std for _, std in gts]) / np.sqrt(len(gts))
    )
    plt.figure(figsize=(10, 5))
    steps = args.batch_size * np.arange(len(preds[0]))
    for (mean, std), label in zip([preds, gts], ["Surrogate", "Oracle"]):
        plt.plot(steps, mean, label=label)
        plt.fill_between(steps, mean - std, mean + std, alpha=0.1)
    plt.xlabel("Optimization Steps")
    plt.ylabel("Warfarin-Associated Dosage Cost")
    plt.xlim(np.min(steps), np.max(steps))
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
    main()
