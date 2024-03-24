#!/usr/bin/env python3
"""
Main driver program for the baseline BOBYQA MBO method.

Author(s):
    Michael Yao @michael-s-yao

Licensed under the MIT License. Copyright University of Pennsylvania 2024.
"""
import argparse
import nlopt
import numpy as np
import os
import sys
import torch
from functools import partial
from pathlib import Path
from typing import Optional, Union

sys.path.append(".")
import mbo
import design_bench
from mbo.run_gabo import load_vae_and_surrogate_models
from models.module import NumPyFCNN
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
        help="Number of algorithm iterations per sample. Default 128."
    )
    parser.add_argument(
        "--solver-samples",
        type=int,
        default=16,
        help="Number of initial starting points to use. Default 16."
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
    solver_steps: int = 128,
    solver_samples: int = 16,
    device: torch.device = torch.device("cpu"),
    seed: int = 42
) -> None:
    """
    Train a surrogate objective model and use BOBYQA to solve a
    Model-Based Optimization (MBO) task.
    Input:
        task_name: name of the offline model-based optimization (MBO) task.
        logging_dir: directory to log the optimization results to.
        step_size: step size for the optimization algorithm. Default 0.01.
        solver_steps: number of iterations per sample. Default 128.
        solver_samples: number initial starting points to use. Default 16.
        device: device. Default CPU.
        seed: random seed. Default 42.
    Returns:
        None.
    """
    task = design_bench.make(task_name)
    nlopt.srand(seed)

    # Task-specific setup.
    if task_name == os.environ["CHEMBL_TASK"]:
        task.map_normalize_y()

    vae, surrogate = load_vae_and_surrogate_models(
        task, task_name, device=device
    )
    _param = next(iter(surrogate.parameters()))
    surrogate = NumPyFCNN(surrogate)

    if task_name in mbo.CONDITIONAL_TASKS:
        start_idxs = np.random.RandomState(seed=seed).choice(
            task.y.shape[0], size=solver_samples, replace=False
        )
    else:
        start_idxs = np.flip(np.argsort(task.y.squeeze()))
        start_idxs = start_idxs[:solver_samples]
    all_x = task.x[start_idxs]
    all_z = []
    for x in all_x:
        all_z.append(
            vae.encode(torch.from_numpy(x[np.newaxis]).to(_param))[0]
        )
    all_z = torch.cat(all_z, dim=0)
    z_shape = tuple(all_z.size())[1:]
    all_z = all_z.reshape(all_z.size(dim=0), -1).detach().cpu().numpy()

    if task_name == os.environ["WARFARIN_TASK"]:
        bounds = task.dataset.opt_dim_bounds
    elif task_name == os.environ["BRANIN_TASK"]:
        bounds = task.oracle.oracle.oracle.bounds
        bounds = bounds.detach().cpu().numpy()
    else:
        inf = 1.0 / np.finfo(np.float16).eps
        bounds = np.vstack([
            -inf * np.ones(vae.latent_size, dtype=np.float32),
            inf * np.ones(vae.latent_size, dtype=np.float32)
        ])

    designs, preds = [], []

    def unit_forward(
        z: np.ndarray,
        grad: np.ndarray,
        conditions: Optional[np.ndarray] = None
    ) -> float:
        assert z.ndim == 1
        if conditions is not None:
            z = np.hstack([z, conditions])
        designs.append(z[np.newaxis])
        y = surrogate(z).item()
        preds.append(y)
        return -1.0 * y

    for zidx in range(all_z.shape[0]):
        if task_name in mbo.CONDITIONAL_TASKS:
            _z = all_z[zidx][np.where(task.dataset.grad_mask)[0]]
            conditions = all_z[zidx][
                np.where(np.logical_not(task.dataset.grad_mask))[0]
            ]
        else:
            _z, conditions = all_z[zidx], None
        opt = nlopt.opt(nlopt.LN_BOBYQA, len(_z))
        opt.set_min_objective(partial(unit_forward, conditions=conditions))
        opt.set_lower_bounds(bounds[0])
        opt.set_upper_bounds(bounds[-1])
        opt.set_maxeval(solver_steps)
        init_size = len(designs)
        try:
            opt.optimize(_z)
        except nlopt.RoundoffLimited:
            pass
        preds += [surrogate(all_z[zidx]).item()] * (
            solver_steps - (len(designs) - init_size)
        )
        designs += [all_z[zidx][np.newaxis]] * (
            solver_steps - (len(designs) - init_size)
        )
    designs = np.concatenate(designs).reshape(
        solver_samples, solver_steps, *z_shape
    )
    scores = [
        task.predict(vae.sample(z=z).detach().cpu().numpy())[np.newaxis]
        for z in torch.from_numpy(designs).to(_param)
    ]
    scores = np.concatenate(scores)
    preds = np.array(preds).reshape(solver_samples, solver_steps, 1)
    if task_name in mbo.CONDITIONAL_TASKS:
        designs = designs.transpose(1, 0, 2)
        preds = preds.transpose(1, 0, 2)
        scores = scores.transpose(1, 0, 2)

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
    run_BOBYQA(**args)


if __name__ == "__main__":
    main()
