"""
Main driver program for toy Branin task experiments.

Author(s):
    Michael Yao

Licensed under the MIT License. Copyright University of Pennsylvania 2023.
"""
import argparse
import numpy as np
import pickle
import sys
import torch
import warnings
from botorch.exceptions.warnings import BadInitialCandidatesWarning
from math import ceil
from pathlib import Path
from typing import Optional, Union

sys.path.append(".")
from branin.objective import Oracle, from_sklearn_model
from branin.policy import GABOPolicy
from experiment.utility import seed_everything


def sample(ref_dataset: torch.Tensor, n: int) -> torch.Tensor:
    """
    Samples a random minibatch from a reference dataset.
    Input:
        ref_dataset: reference dataset of shape Nx2 to sample from.
        n: number of datums from the reference dataset to return.
    Returns:
        A random minibatch of shape nx2 from the reference dataset.
    """
    idxs = torch.randint(
        ref_dataset.size(dim=0), (min(ref_dataset.size(dim=0), n),)
    )
    return ref_dataset[idxs]


def main(
    alpha: Optional[float] = None,
    surrogate: Union[Path, str] = "./branin/ckpts/surrogate.pkl",
    batch_size: int = 16,
    budget: int = 128,
    savepath: Optional[Union[Path, str]] = None,
    seed: int = 42
) -> None:
    """
    Main driver function for toy Branin task experiments.
    Input:
        alpha: optional constant value for alpha.
        surrogate: file path to trained surrogate objective function.
        batch_size: batch size. Default 8.
        budget: total sampling budget. Default 64.
        savepath: optional path to save the results to. Default not saved.
        seed: random seed. Default 42.
    Returns:
        None.
    """
    oracle = Oracle()
    surrogate = from_sklearn_model(surrogate)
    ref_dataset = torch.from_numpy(surrogate.ref_dataset)
    policy = GABOPolicy(
        surrogate,
        alpha=alpha,
        seed=seed,
        x1_range=oracle.x1_range,
        x2_range=oracle.x2_range
    )

    X, y = policy.generate_initial_data(batch_size)
    y_gt = oracle(
        X[:, 0].detach().cpu().numpy(), X[:, 1].detach().cpu().numpy()
    )
    history = []
    policy.fit_critic(sample(ref_dataset, n=(8 * batch_size)), X)
    for step in range(ceil(budget / batch_size) - 1):
        model = policy.fit(X.detach(), y.detach())
        new_X, new_y = policy(model, batch_size, y)
        X_ref = sample(ref_dataset, n=(8 * batch_size))
        alpha = policy.alpha(X_ref)
        history.append(alpha)
        new_y = ((1.0 - alpha) * new_y) - alpha * (
            torch.unsqueeze(policy.wasserstein(X_ref, new_X), dim=-1)
        )
        new_y_gt = oracle(
            new_X[:, 0].detach().cpu().numpy(),
            new_X[:, 1].detach().cpu().numpy()
        )
        X = torch.cat((X, new_X), dim=0)
        y = torch.cat((y, new_y), dim=0)
        y_gt = np.concatenate((y_gt, new_y_gt))
        if step % 1 == 0:
            policy.fit_critic(sample(ref_dataset, n=(8 * batch_size)), X)
    y, y_gt = torch.squeeze(y, dim=-1), torch.from_numpy(y_gt).to(y)

    if savepath is not None:
        with open(savepath, "wb") as f:
            results = {
                "X": X.detach().cpu().numpy(),
                "y": y.detach().cpu().numpy(),
                "y_gt": y_gt.detach().cpu().numpy(),
                "alpha": np.array(history),
                "seed": seed,
                "batch_size": batch_size,
                "budget": budget
            }
            pickle.dump(results, f)

    idxs = torch.argsort(y)
    print(
        f"Best Surrogate: {y[idxs][-1].item():.3f} |",
        f"(Oracle: {y_gt[idxs][-1].item():.3f})"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Branin Toy Experiments")
    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="Optional constant value for alpha."
    )
    parser.add_argument(
        "--savepath",
        type=str,
        default=None,
        help="Optional path to save the optimization results to."
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed. Default 42."
    )
    args = parser.parse_args()

    seed_everything(seed=args.seed)
    torch.set_default_dtype(torch.float64)
    warnings.filterwarnings("ignore", category=BadInitialCandidatesWarning)

    main(
        alpha=args.alpha,
        seed=args.seed,
        savepath=args.savepath,
        surrogate=f"./branin/ckpts/surrogate_{args.seed}.pkl"
    )
