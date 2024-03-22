#!/usr/bin/env python3
"""
Main driver program for baseline L-BFGS MBO method.

Author(s):
    Michael Yao @michael-s-yao

Licensed under the MIT License. Copyright University of Pennsylvania 2024.
"""
import argparse
import numpy as np
import os
import sys
import torch
from pathlib import Path
from scipy.optimize import minimize
from tqdm import tqdm
from typing import Optional, Union

sys.path.append(".")
import mbo  # noqa
import design_bench
from mbo.run_gabo import load_vae_and_surrogate_models
from helpers import seed_everything


def build_args() -> argparse.Namespace:
    """
    Defines the experimental arguments for the baseline gradient ascent MBO
    experiments.
    Input:
        None.
    Returns:
        A namespace with the experimental argument parameters.
    """
    parser = argparse.ArgumentParser(
        description="Baseline L-BFGS MBO Method"
    )

    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=[task.task_name for task in design_bench.registry.all()],
        help="The name of the design-bench task for the experiment."
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed. Default 42."
    )
    parser.add_argument(
        "--logging-dir",
        type=str,
        default="./db-results",
        help="Logging directory. Default `./db-results`"
    )
    parser.add_argument(
        "--solver-steps",
        type=int,
        default=128,
        help="Number of algorithm iterations. Default 128."
    )
    parser.add_argument(
        "--solver-samples",
        type=int,
        default=16,
        help="Number of final designs to return. Default 16."
    )

    return parser.parse_args()


def run_LBFGS(
    task_name: str,
    logging_dir: Optional[Union[Path, str]] = None,
    step_size: float = 0.01,
    solver_steps: int = 128,
    solver_samples: int = 16
) -> None:
    """
    Train a surrogate objective model and perform L-BFGS to solve a
    Model-Based Optimization (MBO) task.
    Input:
        task_name: name of the offline model-based optimization (MBO) task.
        logging_dir: directory to log the optimization results to.
        step_size: step size for the optimization algorithm. Default 0.01.
        solver_steps: number of iterations using the trained surrogate model.
            Default 128.
        solver_samples: total number of final designs to return. Default 16.
    Returns:
        None.
    """
    task = design_bench.make(task_name)

    # Task-specific setup.
    if task_name == os.environ["CHEMBL_TASK"]:
        task.map_normalize_y()

    vae, surrogate = load_vae_and_surrogate_models(task, task_name)
    _param = next(iter(surrogate.parameters()))

    start_idxs = np.argsort(task.y.squeeze())[-solver_samples:]
    all_z = vae.encode(torch.from_numpy(task.x[start_idxs]).to(_param))[0]
    z_shape = all_z.shape[1:]
    all_z = all_z.reshape(solver_samples, -1)
    all_z = all_z.detach().cpu().numpy()[np.newaxis]

    def forward(z: np.ndarray) -> np.ndarray:
        """
        Implements the offline surrogate forward function to *minimize*.
        Input:
            z: input to the forward model.
        Returns:
            The surrogate function evaluated for each input in the batch.
        """
        return -1.0 * (
            surrogate(torch.from_numpy(z).to(_param)).detach().cpu().numpy()
        )

    for _ in tqdm(range(solver_steps), desc="Running L-BFGS"):
        z_step = []
        for zidx in range(solver_samples):
            optimization_step = minimize(
                forward,
                all_z[-1, zidx],
                method="L-BFGS-B",
                options={"maxiter": 1}
            )
            z_step.append(optimization_step.x[np.newaxis])
        z_step = np.concatenate(z_step, axis=0)[np.newaxis]
        all_z = np.concatenate((all_z, z_step), axis=0)
    preds = [-1.0 * forward(z)[np.newaxis] for z in all_z]
    preds = np.concatenate(preds, axis=0)
    all_z = torch.from_numpy(all_z).to(_param)
    if task.is_discrete:
        all_X = torch.vstack([
            vae.sample(z=all_z[opt_step].reshape(-1, *z_shape)).unsqueeze(0)
            for opt_step in range(all_z.shape[0])
        ])
    else:
        all_X = all_z
    scores = np.concatenate([
        task.predict(x.detach().cpu().numpy())[np.newaxis] for x in all_X
    ])
    all_X = all_X.detach().cpu().numpy()

    # Save the optimization results.
    if logging_dir is not None:
        os.makedirs(logging_dir, exist_ok=True)
        np.save(os.path.join(logging_dir, "solution.npy"), all_X)
        np.save(os.path.join(logging_dir, "predictions.npy"), preds)
        np.save(os.path.join(logging_dir, "scores.npy"), scores)


def main():
    args = vars(build_args())
    seed_everything(seed=args.pop("seed"))
    args["task_name"] = args.pop("task")
    run_LBFGS(**args)


if __name__ == "__main__":
    main()
