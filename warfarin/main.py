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
import warnings
from scipy.stats import ttest_rel
from tqdm import tqdm

sys.path.append(".")
warnings.warn = lambda *args, **kwargs: None
from warfarin.cost import dosage_cost
from warfarin.correction import ObjectiveCorrection
from warfarin.dataset import WarfarinDataset
from warfarin.dosing import WarfarinDose
from warfarin.sampler import RandomSampler
from warfarin.transform import PowerNormalizeTransform
from warfarin.metrics import Divergence, SupportCoverage
from models.lipschitz import FrozenMLPRegressor, Lipschitz
from models.anneal import build_schedule
from models.fcnn import FCNN
from models.critic import WeightClipper
from experiment.utility import get_device


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
        type=str,
        required=True,
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
        "--cost_lipschitz_mode",
        type=str,
        choices=("local", "global", None),
        default=None,
        help="Algorithm for calculating the Lipschitz constant of the cost."
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
        "--device", type=str, default="auto", help="Device for training."
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed. Default 42."
    )

    return parser.parse_args()


def main():
    args = build_args()
    device = get_device(args.device)
    rng = np.random.RandomState(args.seed)

    dataset = WarfarinDataset(seed=args.seed)
    dose = WarfarinDose()
    f = FrozenMLPRegressor(args.surrogate_cost)
    f_lipchitz = Lipschitz(f, args.cost_lipschitz_mode, p=2)
    with open(args.hparams, "rb") as fl:
        hparams = json.load(fl)
        clipper_hparams = hparams["WeightClipper"]
        critic_hparams = hparams["SourceCritic"]
        p = hparams[str(type(f.np_model)).split(".")[-1][:-2]]["p"]

    X_train, pred_dose_train = dataset.train_dataset
    X_test, pred_dose_test = dataset.test_dataset
    gt_dose_train = dose(X_train)
    cost_train = dosage_cost(pred_dose_train, gt_dose_train)
    transform = PowerNormalizeTransform(cost_train, p)

    critic = FCNN(**critic_hparams["model"]).to(device)
    critic_clipper = WeightClipper(**clipper_hparams)
    critic_optimizer = optim.Adam(
        critic.parameters(), **critic_hparams["optimizer"]
    )
    critic_lipschitz = Lipschitz(critic, mode="global")

    col_transforms = {}
    for col in [dataset.height, dataset.weight, dataset.dose]:
        t_col = PowerNormalizeTransform(X_train, p=1, key=col)
        X_train = t_col(X_train)
        X_test = t_col(X_test)
        col_transforms[col] = t_col

    z_range = col_transforms[dataset.dose](np.array([0, args.thresh_max]))
    min_z_dose = max(np.min(z_range), args.min_z_dose)
    max_z_dose = min(np.max(z_range), args.max_z_dose)
    policy = RandomSampler(
        min_z_dose=min_z_dose, max_z_dose=max_z_dose, seed=args.seed
    )
    X, alpha = X_test.copy(), build_schedule(args.alpha)
    df = ObjectiveCorrection(X_test, args.num_in_distribution, rng, device)
    best_cost, best_epoch = 1e12, None
    preds, gts = [], []
    if args.cost_lipschitz_mode is not None:
        preds_f_components, preds_df_components = [], []
    with tqdm(
        range(args.num_epochs), desc="Optimizing Warfarin Dose", leave=False
    ) as pbar:
        for i, _ in enumerate(pbar):
            # Calculate relevant Lipschitz constants.
            with warnings.catch_warnings():
                warnings.simplefilter(action="ignore", category=FutureWarning)
                L = f_lipchitz(
                    torch.from_numpy(X.to_numpy().astype(np.float32))
                )
                K = critic_lipschitz()

            # Sample according to the policy.
            for _ in range(args.batch_size):
                loss = 0.0
                if i > 0:
                    t_X = torch.from_numpy(X.to_numpy().astype(np.float32)).to(
                        device
                    )
                    loss -= 10 * alpha() * torch.mean(critic(t_X)).item()
                X = policy(X)
                loss += (1.0 - alpha()) * (f(X) + df(X, critic, K, L))
                policy.feedback(loss)

            optimal_doses = policy.optimum()
            X[policy.dose_key] = optimal_doses
            policy.reset()

            # Calculate statistics.
            f_component, df_component = f(X), df(X, critic, K, L)
            pred_costs = transform.invert(
                np.squeeze(f_component + df_component)
            )
            if args.cost_lipschitz_mode is not None:
                costs_f_component = transform.invert(np.squeeze(f_component))
                costs_df_component = transform.invert(np.squeeze(df_component))
                preds_f_components.append((
                    np.mean(costs_f_component),
                    np.std(costs_f_component, ddof=1)
                ))
                preds_df_components.append((
                    np.mean(costs_df_component),
                    np.std(costs_df_component, ddof=1)
                ))
            cost = np.mean(pred_costs)
            preds.append((cost, np.std(pred_costs, ddof=1)))

            gt_costs = dosage_cost(
                col_transforms[dataset.dose].invert(X[dataset.dose]),
                dose(X)
            )
            gts.append((np.mean(gt_costs), np.std(gt_costs, ddof=1)))

            if args.patience > 0:
                if cost < best_cost:
                    best_cost, best_epoch = cost, i
                if i - best_epoch > args.patience:
                    break

            # Fit the source critic.
            critic.loss_, num_steps = [], 0
            xp = X_test.to_numpy().astype(np.float32)
            xq = X.to_numpy().astype(np.float32)
            while len(critic.loss_) == 0 or (
                np.argmin(critic.loss_) >=
                len(critic.loss_) - min(args.patience, 1)
            ):
                critic.zero_grad()
                if num_steps > args.max_critic_per_epoch >= 0:
                    break
                rng.shuffle(xp)
                rng.shuffle(xq)
                Dw = critic(
                    torch.from_numpy(xp[:args.batch_size, :]).to(device)
                )
                Dw -= critic(
                    torch.from_numpy(xq[:args.batch_size, :]).to(device)
                )
                Dw = torch.mean(Dw)
                Dw.backward()
                critic_optimizer.step()
                num_steps += 1
            pbar.set_postfix(wasserstein=Dw.item(), cost=cost)
            critic_clipper(critic)
            alpha.step()

    print(ttest_rel(pred_costs, gt_costs))

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
    if args.cost_lipschitz_mode is not None:
        preds_f_components = (
            np.array([mean for mean, _ in preds_f_components]),
            np.array([std for _, std in preds_f_components]) / np.sqrt(
                len(X_test)
            )
        )
        preds_df_components = (
            np.array([mean for mean, _ in preds_df_components]),
            np.array([std for _, std in preds_df_components]) / np.sqrt(
                len(X_test)
            )
        )
    plt.figure(figsize=(10, 5))
    steps = args.batch_size * np.arange(len(preds[0]))
    for (mean, sem), label in zip([preds, gts], ["Surrogate", "Oracle"]):
        plt.plot(steps, mean, label=label)
        plt.fill_between(steps, mean - sem, mean + sem, alpha=0.1)
    if args.cost_lipschitz_mode is not None and False:
        for (mean, sem), label in zip(
            [preds_f_components, preds_df_components],
            ["Surrogate (Uncorrected)", "Surrogate (Correction Term)"]
        ):
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
    main()
