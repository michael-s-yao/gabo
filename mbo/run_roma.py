#!/usr/bin/env python3
"""
Main driver program for baseline Robust Model Adaptation (RoMA) MBO method.

Author(s):
    Michael Yao @michael-s-yao

Citation(s):
    [1] Yu S, Ahn S, Song L, Shin J. RoMA: Robust model adaptation for offline
        model-based optimization. Proc NeurIPS. (2021).
        https://openreview.net/forum?id=VH0TRmnqUc

Adapted from the RoMA GitHub repo from @sihyun-yu at https://github.com/
sihyun-yu/RoMA

Licensed under the MIT License. Copyright University of Pennsylvania 2023.
"""
import argparse
import os
import numpy as np
import sys
import tensorflow as tf
from copy import deepcopy
from pathlib import Path
from typing import Optional, Union

sys.path.append(".")
sys.path.append("RoMA")
import mbo
import design_bench
from design_baselines.data import StaticGraphTask
from RoMA.design_baselines.safeweight.nets import DoubleheadModel
from RoMA.design_baselines.safeweight.trainers import Trainer
from RoMA.design_baselines.utils import cont_noise
from helpers import seed_everything


def safeweight(
    task_name: str,
    logging_dir: Optional[Union[Path, str]] = "./db-results",
    batch_size: int = 128,
    val_size: int = 200,
    num_updates: int = 500,
    model_hidden_size: int = 64,
    model_lr: float = 1e-3,
    sol_x_samples: int = 256,
    sol_x_lr: float = 2e-3,
    discrete_smoothing: float = 0.4,
    continuous_noise_std: float = 0.2,
    inner_lr: float = 5e-4,
    region: float = 4.0,
    alpha: float = 1.0,
    warmup_epochs: int = 100
) -> None:
    """
    Implements Robust Model Adaptation (RoMA) to solve a Model-Based
    Optimization (MBO) task. This method is adapted directly from the
    safeweight() and safeweight_latent() baseline methods from the RoMA repo
    cited at the top of this source code file.
    Input:
        task_name: name of the offline model-based optimization (MBO) task.
        logging_dir: directory to log the optimization results to.
        batch_size: the number of samples to load in every batch when drawing
            samples from the training and validation sets. Default 128.
        val_size: the number of samples randomly chosen to be in the validation
            set. Default 200.
        model_hidden_size: model hidden dimensions. Default 64.
        model_lr: learning rate for training the model. Default 1e-3.
        sol_x_samples: number of candidate designs to generate. Default 256.
        sol_x_lr: step size for candidate design generation. Default 2e-3.
        discrete_smoothing: discrete smoothing of the one-hot encoded inputs
            for discrete tasks.
        continuous_noise_std: standard deviation of the continuous noise added
            to the input.
        inner_lr: inner learning rate for the trainer. Default 5e-4.
        region: region specification for the trainer. Default 4.0.
        alpha: regularization parameter for the trainer. Default 1.0.
        warmup_epochs: number of warmup epochs. Default 100.
    Returns:
        None.
    """
    task = StaticGraphTask(task_name, relabel=False)

    if task.wrapped_task.is_discrete:
        ohe = np.full(
            (*task.x.shape, task.wrapped_task.num_classes),
            discrete_smoothing / (task.wrapped_task.num_classes - 1)
        )
        ohe = ohe.astype(np.float32)
        i, j = np.meshgrid(
            np.arange(task.x.shape[0]), np.arange(task.x.shape[1])
        )
        ohe[i.T, j.T, task.x.astype(np.int32)] = 1.0 - discrete_smoothing
        x = np.log(ohe.reshape(ohe.shape[0], -1))
    else:
        x = task.x

    top_idxs = tf.math.top_k(task.y.squeeze(axis=-1), k=sol_x_samples)[1]
    x, y = tf.gather(x, top_idxs, axis=0), tf.gather(task.y, top_idxs, axis=0)
    x, y = x.numpy(), y.numpy()

    trainer = Trainer(
        model=DoubleheadModel(
            input_shape=(x.shape[-1],), hidden=model_hidden_size
        ),
        model_opt=tf.keras.optimizers.Adam(learning_rate=model_lr),
        perturb_fn=(lambda x: cont_noise(x, noise_std=continuous_noise_std)),
        is_discrete=task.wrapped_task.is_discrete,
        sol_x=x,
        sol_y=y,
        sol_x_opt=tf.keras.optimizers.Adam(learning_rate=sol_x_lr),
        coef_stddev=0.0,
        temp_model=DoubleheadModel(
            input_shape=(x.shape[-1],), hidden=model_hidden_size
        ),
        steps_per_update=20,
        mu_x=np.zeros(task.x.shape[0]),
        st_x=np.ones(task.x.shape[0]),
        inner_lr=inner_lr,
        region=region,
        max_y=task.y.max(),
        lr=sol_x_lr,
        alpha=alpha
    )

    train, val = task.build(x, y, batch_size=batch_size, val_size=val_size)
    for _ in range(warmup_epochs):
        for xx, yy in train:
            trainer.train_step(xx, yy)
        for xx, yy in val:
            trainer.validate_step(xx, yy)

    if task_name in mbo.CONDITIONAL_TASKS:
        conditions = deepcopy(x)
        grad_mask = task.wrapped_task.dataset.grad_mask
    for step in range(num_updates):
        trainer.init_step()
        trainer.update_step()
        if task_name in mbo.CONDITIONAL_TASKS:
            trainer.sol_x = tf.Variable(
                (trainer.sol_x * grad_mask) + (conditions * (1.0 - grad_mask))
            )
    x, ypred = trainer.get_sol_x().numpy(), trainer.sol_y.numpy()
    if task.wrapped_task.is_discrete:
        x = x.reshape(sol_x_samples, -1, task.wrapped_task.num_classes)
        x = tf.argmax(tf.math.softmax(tf.convert_to_tensor(x)), axis=-1)
        x = x.numpy().astype(np.int32)
    y = task.wrapped_task.predict(x)

    # Save the optimization results.
    if logging_dir is not None:
        os.makedirs(logging_dir, exist_ok=True)
        np.save(os.path.join(logging_dir, "solution.npy"), x[np.newaxis, ...])
        np.save(
            os.path.join(logging_dir, "predictions.npy"),
            ypred[np.newaxis, ...]
        )
        np.save(os.path.join(logging_dir, "scores.npy"), y[np.newaxis, ...])


def build_args() -> argparse.Namespace:
    """
    Defines the experimental arguments for the baseline RoMA method MBO
    experiments.
    Input:
        None.
    Returns:
        A namespace with the experimental argument parameters.
    """
    parser = argparse.ArgumentParser(description="Baseline RoMA Method MBO")

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
        "--budget", type=int, default=256, help="Query budget. Default 256."
    )

    return parser.parse_args()


def main():
    args = build_args()
    seed_everything(args.seed)
    safeweight(args.task, args.logging_dir, sol_x_samples=args.budget)


if __name__ == "__main__":
    main()
