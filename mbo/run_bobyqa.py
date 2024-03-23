#!/usr/bin/env python3
"""
Main driver program for the baseline BOBYQA MBO method.

Author(s):
    Michael Yao @michael-s-yao

Licensed under the MIT License. Copyright University of Pennsylvania 2024.
"""
import argparse
import numpy as np
import pybobyqa
import os
import sys
import torch
from pathlib import Path
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
        "--maxiter",
        type=int,
        default=2048,
        help="Maximum number of total algorithm iterations. Default 2048."
    )
    parser.add_argument(
        "--device",
        type=torch.device,
        default="cpu",
        help="Device. Default CPU."
    )

    return parser.parse_args()


def run_BOBYQA(
    task_name: str,
    logging_dir: Optional[Union[Path, str]] = None,
    step_size: float = 0.01,
    maxiter: int = 2048,
    device: torch.device = torch.device("cpu")
) -> None:
    """
    Train a surrogate objective model and use BOBYQA to solve a
    Model-Based Optimization (MBO) task.
    Input:
        task_name: name of the offline model-based optimization (MBO) task.
        logging_dir: directory to log the optimization results to.
        step_size: step size for the optimization algorithm. Default 0.01.
        maxiter: maximum number of iterations. Default 2048.
        device: device. Default CPU.
    Returns:
        None.
    """
    task = design_bench.make(task_name)

    # Task-specific setup.
    if task_name == os.environ["CHEMBL_TASK"]:
        task.map_normalize_y()

    vae, surrogate = load_vae_and_surrogate_models(
        task, task_name, device=device
    )
    _param = next(iter(surrogate.parameters()))

    start_idxs = np.flip(np.argsort(task.y.squeeze()))
    start_idxs = start_idxs[:min(maxiter, task.y.shape[0])]
    all_x = task.x[start_idxs]
    all_z = []
    for x in all_x:
        all_z.append(
            vae.encode(torch.from_numpy(x[np.newaxis]).to(_param))[0]
        )
    all_z = torch.cat(all_z, dim=0)
    z_shape = tuple(all_z.size())[1:]
    all_z = all_z.reshape(all_z.size(dim=0), -1).detach().cpu().numpy()

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

    if task_name == os.environ["WARFARIN_TASK"]:
        bounds = torch.vstack([
            torch.zeros(vae.latent_size).to(_param),
            torch.ones(vae.latent_size).to(_param)
        ])
        bounds = torch.hstack([
            torch.from_numpy(task.dataset.opt_dim_bounds).to(_param),
            bounds
        ])
        bounds = bounds[:, :vae.latent_size]
        for cvar in task.dataset.continuous_vars:
            bounds[0, task.dataset.column_names.tolist().index(cvar)] = -10.0
            bounds[1, task.dataset.column_names.tolist().index(cvar)] = 10.0
        bounds = bounds.detach().cpu().numpy()
    elif task_name == os.environ["BRANIN_TASK"]:
        bounds = task.oracle.oracle.oracle.bounds
        bounds = bounds.detach().cpu().numpy()
    else:
        inf = (1.0 / np.finfo(np.float32).eps)
        bounds = np.vstack([
            -inf * np.ones(vae.latent_size, dtype=np.float32),
            inf * np.ones(vae.latent_size, dtype=np.float32)
        ])

    solz = []
    for zidx in range(all_z.shape[0]):
        solution = pybobyqa.solve(
            forward,
            all_z[zidx],
            bounds=(bounds[0], bounds[-1]),
            seek_global_minimum=True,
            do_logging=True,
            user_params={
                "logging.save_xk": True, "logging.save_diagnostic_info": True
            }
        )
        for xk in solution.diagnostic_info["xk"].iloc[::-1]:
            solz.append(xk[np.newaxis])
        if len(solz) >= maxiter:
            break
    all_z = np.concatenate(solz, axis=0)[:maxiter]

    preds = -1.0 * forward(all_z)[np.newaxis]
    all_z = all_z.reshape(all_z.shape[0], *z_shape)
    all_z = torch.from_numpy(all_z).to(_param)
    scores = task.predict(
        vae.sample(z=all_z.reshape(-1, *z_shape)).detach().cpu().numpy()
    )
    scores = scores[np.newaxis]
    all_z = all_z.detach().cpu().numpy()[np.newaxis]

    # Save the optimization results.
    if logging_dir is not None:
        os.makedirs(logging_dir, exist_ok=True)
        np.save(os.path.join(logging_dir, "solution.npy"), all_z)
        np.save(os.path.join(logging_dir, "predictions.npy"), preds)
        np.save(os.path.join(logging_dir, "scores.npy"), scores)


def main():
    args = vars(build_args())
    seed_everything(seed=args.pop("seed"))
    args["task_name"] = args.pop("task")
    run_BOBYQA(**args)


if __name__ == "__main__":
    main()
