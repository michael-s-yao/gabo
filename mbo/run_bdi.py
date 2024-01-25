#!/usr/bin/env python3
"""
Main driver program for baseline Bidirectional Learning for Offline Infinite-
width Model-Based Optimization (BDI) MBO method.

Author(s):
    Michael Yao @michael-s-yao

Citation(s):
    [1] Chen C, Zhang Y, Fu J, Liu X, Coates M. Bidirectional learning for
        offline infinite-width model-based optimization. Proc NeurIPS. (2022).
        https://openreview.net/forum?id=_j8yVIyp27Q

Adapted from the BDI GitHub repo from @GGchen1997 at https://github.com/
GGchen1997/BDI

Licensed under the MIT License. Copyright University of Pennsylvania 2023.
"""
import argparse
import functools
import jax
import numpy as np
import os
import scipy
import sys
import torch
import tensorflow as tf
from copy import deepcopy
from jax import numpy as jnp
from jax import scipy as jsp
from jax.experimental import optimizers
from neural_tangents import stax
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, Union

sys.path.append(".")
import mbo
import design_bench
from design_bench.datasets.discrete.chembl_dataset import ChEMBLDataset
from BDI.utils import get_update_functions, load_d, load_weights
from helpers import seed_everything


def load_data(
    task: design_bench.task.Task, task_name: str
) -> Tuple[Union[np.ndarray, Tuple[int]]]:
    """
    Generalized, cleaned-up implementation of the process_data() function from
    https://github.com/GGchen1997/BDI/utils.py. Loads and processes the MBO
    tasks's offline data.
    Input:
        task: an offline model-based optimization (MBO) task.
        task_name: name of the offline MBO task.
    Returns:
        x: the loaded design observations associated with the task.
        y: the corresponding objective values associated with the designs.
        x_shape: the shape of the input processed designs.
    """
    # We have to restrict the number of datums used for calculating k_tt in the
    # UTR task in the compute_d() function below due to OOM errors.
    if task_name == os.environ["UTR_TASK"]:
        idxs = np.argsort(task.y.squeeze(axis=-1))[-(task.x.shape[0] // 2):]
        x, y = task.x[idxs], task.y[idxs]
    else:
        x, y = deepcopy(task.x), task.y

    if task.is_discrete:
        x = task.to_logits(x)

    x_shape = x.shape
    if task_name not in mbo.CONDITIONAL_TASKS:
        x = task.normalize_x(x).reshape(x.shape[0], -1)
        y = task.normalize_y(y)

    return x, y, x_shape


def compute_d(
    task_name: str,
    x: np.ndarray,
    y: np.ndarray,
    reg: float = 1e-6,
    savedir: Optional[Union[Path, str]] = "npy"
) -> jnp.ndarray:
    """
    Implements the `BDI/npy/compute_d.py` script from the cited BDI reference
    codebase cited above.
    Input:
        task_name: name of the offline MBO task.
        x: the observed designs from the offline dataset.
        y: the observed objective values from the offline dataset.
        reg: regularization strength. Default 1e-6.
        savedir: optional savepath to save the result to.
    Returns:
        The solution to the specified linear matrix equation.
    """
    if os.path.isfile(os.path.join(savedir, f"{task_name}.npy")):
        return np.load(os.path.join(savedir, f"{task_name}.npy"))
    init_fn, apply_fn, kernel_fn = stax.serial(
        stax.Dense(1),
        stax.Relu(),
        stax.Dense(1),
        stax.Relu(),
        stax.Dense(1),
        stax.Relu(),
        stax.Dense(1),
        stax.Relu(),
        stax.Dense(1),
        stax.Relu(),
        stax.Dense(1)
    )

    k_tt = functools.partial(kernel_fn, get="ntk")(x, x)
    k_tt_reg = k_tt + np.abs(reg) * np.trace(k_tt) * np.eye(k_tt.shape[0]) / (
        k_tt.shape[0]
    )
    d = scipy.linalg.solve(k_tt_reg, y, sym_pos=True)
    if savedir is not None:
        os.makedirs(savedir, exist_ok=True)
        np.save(os.path.join(savedir, f"{task_name}.npy"), d)
    return d


def make_loss_fn(
    kernel_fn: Callable, d: np.ndarray, idx: int, reg: float = 0.0
) -> Callable:
    """
    Returns the loss function per Equation (14) of Chen et al. (2022) cited
    above but adapted for conditional MBO tasks.
    Input:
        kernel_fn: the neural kernel function.
        d: the vector of solutions to the linear matrix equation from
            `compute_d()` above.
        idx: the index of the element from the dataset to consider.
        reg: regularization strength. Default 0.0.
    Returns:
        The specified loss function.
    """
    @jax.jit
    def conditional_loss_fn_both(
        x_support: jnp.ndarray,
        y_support: jnp.ndarray,
        x_target: jnp.ndarray,
        y_target: jnp.ndarray
    ) -> Tuple[jnp.ndarray]:
        y_support = jax.lax.stop_gradient(y_support)
        k_ss = kernel_fn(x_support, x_support)
        k_ts = kernel_fn(x_target, x_support)
        k_ss_reg = (
            k_ss + jnp.abs(reg) * jnp.trace(k_ss) * jnp.eye(k_ss.shape[0]) / (
                k_ss.shape[0]
            )
        )
        pred = jnp.dot(
            k_ts, jsp.linalg.solve(k_ss_reg, y_support, sym_pos=True)
        )
        mse_loss1 = 0.5 * jnp.sum((pred - y_target) ** 2)
        k_st = kernel_fn(x_support, x_target)
        pred = jnp.dot(k_st, d[idx])
        mse_loss2 = 0.5 * jnp.mean((pred - y_support) ** 2)
        mse_loss = mse_loss1 + mse_loss2
        return mse_loss, mse_loss

    return conditional_loss_fn_both


def get_conditional_update_functions(
    init_params: Dict[str, np.ndarray],
    kernel_fn: Callable,
    d: np.ndarray,
    idx: int,
    lr: float = 0.1,
    reg: float = 0.0
) -> Tuple[Any]:
    """
    Implements the `get_update_functions()` in the source code file
    BDI/utils.py in the BDI repository cited above for conditional MBO
    tasks.
    Input:
        init_params: a dictionary containing the initial x and y values.
        kernel_fn: the neural kernel function.
        lr: learning rate. Default 0.1.
        d: the vector of solutions to the linear matrix equation from
            `compute_d()` above.
        idx: the index of the element from the dataset to consider.
        reg: regularization strength. Default 0.0.
    Returns:
        opt_state, get_params, update_fn.
    """
    opt_init, opt_update, get_params = optimizers.adam(lr)
    opt_state = opt_init(init_params)
    conditional_loss_fn = make_loss_fn(kernel_fn, d, idx, reg=reg)
    grad_loss = jax.grad(
        lambda params, x_target, y_target: conditional_loss_fn(
            params["x"], params["y"], x_target, y_target
        ),
        has_aux=True
    )

    @jax.jit
    def update_fn(step, opt_state, params, x_target, y_target):
        dparams, aux = grad_loss(params, x_target, y_target)
        return opt_update(step, dparams, opt_state), aux

    return opt_state, get_params, update_fn


def bdi(
    task_name: str,
    gamma: float = 0.0,
    budget: int = 256,
    label: float = 10.0,
    lr: float = 0.1
) -> float:
    """
    Implements Bidirectional Learning for Offline Infinite-width MBO (BDI) to
    solve a Model-Based Optimization (MBO) task. This method is adapted
    directly from the BDI.py script in the source code of the BDI repo cited at
    the top of this source code file.
    Input:
        task_name: name of the offline model-based optimization (MBO) task.
        gamma: gamma parameter for loss weighting. Default 0.0.
        budget: model evaluation budget. Default 256.
        label: label value for the y values. Default 10.0
        lr: learning rate. Default 0.1.
    Returns:
        The oracle score of the final proposed design candidate.
    """
    task = design_bench.make(task_name)
    task_x, task_y, x_shape = load_data(task, task_name)

    d = compute_d(task_name, task_x, task_y)
    load_d(task_name)
    load_weights(task_name, task_y, gamma=gamma)

    if task_name not in mbo.CONDITIONAL_TASKS:
        x = deepcopy(task_x[np.argmax(task_y.squeeze(axis=-1))])
    y = label * np.ones((1, 1))

    init_fn, apply_fn, kernel_fn = stax.serial(
        stax.Dense(1),
        stax.Relu(),
        stax.Dense(1),
        stax.Relu(),
        stax.Dense(1),
        stax.Relu(),
        stax.Dense(1),
        stax.Relu(),
        stax.Dense(1),
        stax.Relu(),
        stax.Dense(1)
    )
    kernel_fn = functools.partial(kernel_fn, get="ntk")

    if task_name in mbo.CONDITIONAL_TASKS:
        opt_state, params, update_fn = [], [], []
        for idx, (xx, yy) in enumerate(zip(task_x, task_y)):
            state, get_params, updt_fn_ = (
                get_conditional_update_functions(
                    {"x": xx.reshape(1, -1), "y": yy}, kernel_fn, d, idx, lr
                )
            )
            opt_state.append(state)
            params.append(get_params(state))
            update_fn.append(updt_fn_)
    else:
        opt_state, get_params, update_fn = get_update_functions(
            {"x": x.reshape(1, -1), "y": y},
            kernel_fn,
            lr,
            mode="both"
        )
        params = get_params(opt_state)
    x_targ, y_targ = task_x, task_y

    if task_name in mbo.CONDITIONAL_TASKS:
        x = deepcopy(x_targ)
        grad_mask = task.dataset.grad_mask
    else:
        x = deepcopy(x_targ[0])[np.newaxis, :]
        x_targ, y_targ = x_targ[np.newaxis, ...], y_targ[np.newaxis, ...]
    for i, (xx, yy) in enumerate(zip(x_targ, y_targ)):
        if task_name in mbo.CONDITIONAL_TASKS:
            state_, params_, update_fn_ = opt_state[i], params[i], update_fn[i]
            conditions = deepcopy(params_["x"])
        else:
            state_, params_, update_fn_ = opt_state, params, update_fn

        for step in range(budget):
            if task_name in mbo.CONDITIONAL_TASKS:
                state_, train_loss = update_fn_(
                    step + 1,
                    state_,
                    params_,
                    xx[np.newaxis],
                    yy[np.newaxis],
                )
            else:
                state_, train_loss = update_fn_(
                    step + 1, state_, params_, xx, yy
                )
            params_ = get_params(state_)
            if task_name in mbo.CONDITIONAL_TASKS:
                params_["x"] = (params_["x"] * grad_mask) + (
                    conditions * (1.0 - grad_mask)
                )
        x[i] = np.array(params_["x"]).squeeze(axis=0)
    x = x.reshape(-1, x_shape[-2], x_shape[-1]) if task.is_discrete else x
    if task_name not in mbo.CONDITIONAL_TASKS:
        x = task.denormalize_x(x)
    if task.is_discrete:
        x = task.to_integers(x)

    if isinstance(x, tf.Tensor):
        x = x.numpy()
    elif isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    elif isinstance(x, jnp.ndarray):
        x = np.array(x)

    y = task.predict(x)
    y = y.item(0) if y.size == 1 else y.squeeze(axis=-1)
    return y


def build_args() -> argparse.Namespace:
    """
    Defines the experimental arguments for the baseline BDI method MBO
    experiments.
    Input:
        None.
    Returns:
        A namespace with the experimental argument parameters.
    """
    parser = argparse.ArgumentParser(description="Baseline BDI Method MBO")

    parser.add_argument(
        "--task", type=str, required=True, help="Name of offline MBO task."
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 43, 44, 45, 46],
        help="Random seeds."
    )

    return parser.parse_args()


def main():
    args = build_args()

    ys = []
    for seed in args.seeds:
        seed_everything(seed)
        ys.append(bdi(args.task))

    if args.task == os.environ["CHEMBL_TASK"]:
        # Normalize the ChEMBL task results.
        _, standard_type, assay_chembl_id, _ = args.task.split("_")
        y_shards = ChEMBLDataset.register_y_shards(
            assay_chembl_id=assay_chembl_id, standard_type=standard_type
        )
        y = np.vstack([np.load(shard.disk_target) for shard in y_shards])
        ys = (np.array(ys) - y.min()) / (y.max() - y.min())

    print(args.task + ":", np.mean(ys), "+/-", np.std(ys))


if __name__ == "__main__":
    main()
