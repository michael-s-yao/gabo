import argparse
import numpy as np
import os
import sys
import torch
from pathlib import Path
from simanneal import Annealer
from typing import Optional, Union

sys.path.append(".")
import mbo
import design_bench
from mbo.run_gabo import load_vae_and_surrogate_models
from models.module import NumPyFCNN
from helpers import seed_everything


ALL_DESIGNS = []


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


class LoggingList(list):
    def copy(self):
        ALL_DESIGNS.append(self[:])
        return self[:]


class MBOOptimizationProblem(Annealer):
    def __init__(
        self,
        task_name: str,
        task: design_bench.task.Task,
        init_state: np.ndarray,
        conditions: Optional[np.ndarray] = None,
        bounds: Optional[np.ndarray] = None,
        seed: int = 42,
        device: torch.device = torch.device("cpu")
    ):
        """
        Args:
            task_name: name of the offline model-based optimization (MBO) task.
            task: the offline model-based optimization (MBO) task.
            init_state: the initial state to optimize from.
            conditions: an optional array of condition values for conditional
                MBO problems.
            bounds: an optional array of bounds for continuous MBO problems.
                Must be specified for continuous MBO problems. Expect a 2xD
                array if specified, where D is the number of state dimensions.
            seed: random seed. Default 42.
            device: device. Default CPU.
        """
        super(MBOOptimizationProblem, self).__init__(
            LoggingList(init_state.tolist())
        )
        self.task_name = task_name
        self.task = task
        self.conditions = conditions
        self.bounds = bounds
        self.seed = seed
        self.copy_strategy = "method"
        self._rng = np.random.RandomState(seed=seed)
        self.vae, self.surrogate = load_vae_and_surrogate_models(
            self.task, self.task_name, device=device
        )
        _param = next(iter(self.surrogate.parameters()))
        self.torch_config = {"device": _param.device, "dtype": _param.dtype}
        self.surrogate = NumPyFCNN(self.surrogate)

    def move(self) -> None:
        """
        Create a state change.
        Input:
            None.
        Returns:
            None.
        """
        if self.task.is_discrete:
            a, b = self._rng.randint(0, len(self.state), size=2)
            self.state[a], self.state[b] = self.state[b], self.state[a]
        else:
            for dim in range(len(self.state)):
                self.state[dim] = self._rng.uniform(
                    low=self.bounds[0, dim], high=self.bounds[1, dim]
                )

    def energy(self) -> float:
        """
        Computes the energy of a configuration state design.
        Input:
            None.
        Returns:
            The energy of the current configuration state.
        """
        if self.conditions is not None:
            x = np.hstack([self.state, self.conditions]).tolist()
        else:
            x = self.state
        return -1.0 * self.forward(x)

    def forward(self, x: list) -> float:
        """
        Forward pass through the surrogate objective function.
        Input:
            x: a single input design.
        Returns:
            The value of the surrogate objective function at the input design.
        """
        z = self.vae.encode(torch.tensor([x], **self.torch_config))[0]
        z = z.detach().cpu().numpy().flatten()[np.newaxis]
        return self.surrogate(z).item()


def run_simanneal(
    task_name: str,
    logging_dir: Optional[Union[Path, str]] = None,
    solver_steps: int = 128,
    solver_samples: int = 16,
    device: torch.device = torch.device("cpu"),
    seed: int = 42
) -> None:
    """
    Use simulated annealing to solve a Model-Based Optimization (MBO) task.
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
    global ALL_DESIGNS
    task = design_bench.make(task_name)

    # Task-specific setup.
    if task_name == os.environ["CHEMBL_TASK"]:
        task.map_normalize_y()

    if task_name in mbo.CONDITIONAL_TASKS:
        start_idxs = np.random.RandomState(seed=seed).choice(
            task.y.shape[0], size=solver_samples, replace=False
        )
    else:
        start_idxs = np.flip(np.argsort(task.y.squeeze()))
        start_idxs = start_idxs[:solver_samples]
    all_x = task.x[start_idxs]

    if task_name == os.environ["WARFARIN_TASK"]:
        bounds = task.dataset.opt_dim_bounds
    elif task_name == os.environ["BRANIN_TASK"]:
        bounds = task.oracle.oracle.oracle.bounds
        bounds = bounds.detach().cpu().numpy()
    else:
        bounds = None

    for xidx in range(all_x.shape[0]):
        _x, conditions = all_x[xidx], None
        if task_name in mbo.CONDITIONAL_TASKS:
            _x, conditions = _x[np.where(task.dataset.grad_mask)[0]], _x[
                np.where(np.logical_not(task.dataset.grad_mask))[0]
            ]
        problem = MBOOptimizationProblem(
            task_name=task_name,
            task=task,
            init_state=_x,
            conditions=conditions,
            bounds=bounds,
            seed=seed,
            device=device
        )
        problem.set_schedule(problem.auto(minutes=0.2, steps=solver_steps))
        state, _ = problem.anneal()
        ALL_DESIGNS.append(state)
        ALL_DESIGNS += (((xidx + 1) * solver_steps) - len(ALL_DESIGNS)) * [_x]
    designs = np.vstack(ALL_DESIGNS)
    preds = np.array([problem.forward(x.tolist()) for x in designs])
    scores = task.predict(designs)
    designs = designs.reshape(solver_steps, solver_samples, -1)
    preds = preds.reshape(solver_steps, solver_samples, 1)
    scores = scores.reshape(solver_steps, solver_samples, 1)

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
    run_simanneal(**args)


if __name__ == "__main__":
    main()
