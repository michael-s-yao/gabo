#!/usr/bin/env python3
"""
Main driver program for baseline CMA-ES MBO methods.

Author(s):
    Michael Yao @michael-s-yao

Adapted from the design-baselines GitHub repo from @brandontrabucco.
https://github.com/brandontrabucco/design-baselines/design_baselines/
cma_es/__init__.py

Licensed under the MIT License. Copyright University of Pennsylvania 2024.
"""
import argparse
import cma
import numpy as np
import os
import sys
import torch
from pathlib import Path
from typing import Optional, Union

sys.path.append(".")
import mbo
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
        description="Baseline Gradient Ascent MBO"
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
        help="Number of gradient ascent steps to perform per sample."
    )
    parser.add_argument(
        "--solver-samples",
        type=int,
        default=16,
        help="Number of datums to perform gradient ascent from."
    )
    parser.add_argument(
        "--device",
        type=torch.device,
        default="cpu",
        help="Device. Default CPU."
    )

    return parser.parse_args()


def cma_es(
    task_name: str,
    logging_dir: Optional[Union[Path, str]] = None,
    solver_steps: int = 128,
    solver_samples: int = 16,
    cma_sigma: float = 0.5,
    seed: int = 42,
    device: torch.device = torch.device("cpu")
) -> None:
    """
    Perform CMA-ES optimization algorithm to solve a Model-Based Optimization
    (MBO) task. This method is adapted directly from the `cma_es()` baseline
    method from the design-baselines repo cited at the top of this source code
    file.
    Input:
        task_name: name of the offline model-based optimization (MBO) task.
        logging_dir: directory to log the optimization results to.
        solver_steps: number of steps for gradient ascent against the trained
            surrogate model. Default 128.
        solver_samples: total number of final designs to return. Default 16.
        cma_sigma: initial standard deviation for the CMA-ES algorithm.
            Default 0.5.
        seed: random seed. Default 42.
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
        bounds = bounds.detach().cpu().numpy().tolist()
    elif task_name == os.environ["BRANIN_TASK"]:
        bounds = task.oracle.oracle.oracle.bounds
        bounds = bounds.detach().cpu().numpy().tolist()
    else:
        bounds = None

    # Select the top k initial designs from the dataset as starting points.
    if task_name not in mbo.CONDITIONAL_TASKS:
        start_idxs = np.argsort(task.y.squeeze())[-solver_samples:]
    else:
        start_idxs = np.random.RandomState(seed=seed).choice(
            task.y.shape[0], size=solver_samples, replace=False
        )
    all_z = vae.encode(torch.from_numpy(task.x[start_idxs]).to(_param))[0]
    z_shape = all_z.shape[1:]
    if task_name not in mbo.CONDITIONAL_TASKS:
        all_z = all_z.reshape(solver_samples, -1)
    all_z = all_z.detach().cpu().numpy()

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

    designs, preds, scores = [], [], []
    for i in range(solver_samples):
        zi = all_z[i].flatten().tolist()

        es = cma.CMAEvolutionStrategy(
            zi, cma_sigma, {"seed": seed, "bounds": bounds}
        )
        step = 0
        result = []
        while not es.stop() and step < solver_steps:
            solutions = es.ask()
            es.tell(solutions, [forward(z).item() for z in solutions])
            if task_name in mbo.CONDITIONAL_TASKS:
                result.append(
                    (es.result.xbest * task.dataset.grad_mask) + (
                        zi * np.logical_not(task.dataset.grad_mask)
                    )
                )
            else:
                result.append(es.result.xbest)
            step += 1
        result = ([zi] * (solver_steps - len(result))) + result
        designs.append(np.vstack(result)[np.newaxis])
        preds.append(forward(designs[-1].squeeze(axis=0))[np.newaxis])
        _z = torch.from_numpy(
            designs[-1].squeeze(axis=0).reshape(-1, *z_shape)
        )
        x = vae.sample(z=_z.to(_param))
        scores.append(task.predict(x.detach().cpu().numpy())[np.newaxis])
    designs = np.concatenate(designs, axis=0)
    preds = np.concatenate(preds, axis=0)
    scores = np.concatenate(scores, axis=0)

    if task_name not in mbo.CONDITIONAL_TASKS:
        preds = preds.reshape(solver_samples, solver_steps, 1)
        scores = scores.reshape(solver_samples, solver_steps, 1)
    else:
        preds = preds.squeeze(axis=-1).T[..., np.newaxis]
        scores = scores.squeeze(axis=-1).T[..., np.newaxis]

    # Save the optimization results.
    if logging_dir is not None:
        os.makedirs(logging_dir, exist_ok=True)
        np.save(os.path.join(logging_dir, "solution.npy"), designs)
        np.save(os.path.join(logging_dir, "predictions.npy"), preds)
        np.save(os.path.join(logging_dir, "scores.npy"), scores)


def main():
    args = vars(build_args())
    seed_everything(args["seed"])
    args["task_name"] = args.pop("task")
    cma_es(**args)


if __name__ == "__main__":
    main()
