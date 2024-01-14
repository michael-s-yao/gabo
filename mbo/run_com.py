#!/usr/bin/env python3
"""
Main driver program for the COM baseline MBO method.

Author(s):
    Michael Yao @michael-s-yao

Citation(s):
    [1] Trabucco B, Kumar A, Geng X, Levine S. Conservative objective models
        for effective offline model-based optimization. Proc ICML 139:10358-
        68. (2021). http://proceedings.mlr.press/v139/trabucco21a.html

Adapted from the design-baselines GitHub repo from @brandontrabucco at
https://github.com/brandontrabucco/design-baselines/design_baselines/
coms_cleaned/__init__.py

Licensed under the MIT License. Copyright University of Pennsylvania 2023.
"""
import click
import numpy as np
import os
import sys
import tensorflow as tf
from pathlib import Path
from typing import Optional, Union

sys.path.append(".")
import mbo  # noqa
from design_baselines.coms_cleaned import coms_cleaned
from design_baselines.data import StaticGraphTask, build_pipeline
from design_baselines.logger import Logger
from design_baselines.coms_cleaned.nets import ForwardModel
from models.ccom import ConditionalConservativeObjectiveModel
from helpers import seed_everything


def pop_seed() -> Optional[int]:
    """
    Finds, returns, and removes the random seed (if specified) from the
    command line arguments.
    Input:
        None.
    Returns:
        The random seed value (if specified).
    """
    for i in range(len(sys.argv) - 1):
        if sys.argv[i] == "--seed":
            seed = int(sys.argv[i + 1])
            sys.argv = sys.argv[:i] + sys.argv[(i + 2):]
            return int(seed)
    return None


def get_task_name() -> Optional[str]:
    """
    Finds and returns the model-based optimization (MBO) task name from the
    command line arguments.
    Input:
        None.
    Returns:
        The task name (if found).
    """
    for i in range(len(sys.argv) - 1):
        if sys.argv[i] == "--task":
            return str(sys.argv[i + 1])
    return None


@click.command()
@click.option(
    "--task",
    type=str,
    required=True,
    help="The name of the model-based optimization (MBO) task."
)
@click.option(
    "--logging-dir",
    default="coms-cleaned",
    type=str,
    help="Tensorboard logging directory."
)
@click.option(
    "--particle-train-gradient-steps",
    default=256,
    type=int,
    help="Number of steps used in the COMs inner loop for training."
)
@click.option(
    "--particle-evaluate-gradient-steps",
    default=256,
    type=int,
    help="Number of steps used in the COMs inner loop for evaluation."
)
@click.option(
    "--evaluation-samples",
    default=1,
    type=int,
    help="Number of evaluation samples. Default 1."
)
@click.option(
    "--fast/--not-fast",
    default=False,
    type=bool,
    help="Whether to run the experiment quickly and log only once."
)
def coms_partial(
    task: str,
    logging_dir: Union[Path, str],
    particle_lr: float = 0.01,
    particle_train_gradient_steps: int = 50,
    particle_evaluate_gradient_steps: int = 256,
    batch_size: int = 128,
    val_size: int = 200,
    num_epochs: int = 100,
    forward_model_lr: float = 0.0003,
    forward_model_alpha: float = 1.0,
    forward_model_alpha_lr: float = 0.01,
    forward_model_overestimation_limit: float = 0.5,
    **kwargs
) -> None:
    """
    Solve a continuous Model-Based Optimization (MBO) problem using the
    baseline Conservative Objective Models (COMs) method over a subset of the
    input design dimensions.
    Input:
        task: the name of the MBO task.
        logging_dir: the directory in which to log the optimization results.
        particle_lr: the learning rate for the COMs inner loop.
        particle_train_gradient_steps: the number of gradient ascent steps used
            in the COMs inner loop for training.
        particle_evaluate_gradient_steps: the number of gradient ascent steps
            used in the COMs inner loop for evaluation.
        batch_size: batch size to use when training the forward model.
        val_size: size of the validation set for training the forward model.
        num_epochs: number of training epochs for the forward model.
        forward_model_lr: learning rate of the forward model.
        forward_model_alpha: the initial Lagrange multiplier of the forward
            model.
        forward_model_alpha_lr: the learning rate of the Lagrange multiplier.
        forward_model_overestimation_limit: target for tuning the Lagrange
            multiplier.
    Returns:
        None.
    """
    logger = Logger(logging_dir)

    # Create a model-based optimization task.
    task = StaticGraphTask(task, relabel=False)

    # make a neural network to predict scores
    forward_model = ForwardModel(
        task.input_shape,
        activations=("leaky_relu", "leaky_relu"),
        hidden_size=2048,
        final_tanh=False
    )

    # Compute the normalized learning rate of the model.
    particle_lr = particle_lr * np.sqrt(
        np.sum(task.wrapped_task.dataset.grad_mask)
    )

    train, validate = build_pipeline(
        x=task.x,
        y=task.y,
        batch_size=batch_size,
        val_size=val_size,
        bootstraps=0
    )
    trainer = ConditionalConservativeObjectiveModel(
        grad_mask=task.wrapped_task.dataset.grad_mask,
        forward_model=forward_model,
        forward_model_opt=tf.keras.optimizers.Adam,
        forward_model_lr=forward_model_lr,
        alpha=forward_model_alpha,
        alpha_opt=tf.keras.optimizers.Adam,
        alpha_lr=forward_model_alpha_lr,
        overestimation_limit=forward_model_overestimation_limit,
        particle_lr=particle_lr,
        noise_std=0.0,
        particle_gradient_steps=particle_train_gradient_steps,
        entropy_coefficient=0.0
    )
    trainer.launch(train, validate, logger=logger, epochs=num_epochs)

    X = tf.concat([x for x, _ in validate], axis=0)
    all_X = X.numpy()[np.newaxis, ...]

    scores, preds = [], []
    for _ in range(particle_evaluate_gradient_steps):
        X = trainer.optimize(X, steps=1, training=False)
        all_X = np.concatenate([all_X, X.numpy()[np.newaxis, ...]], axis=0)
        scores.append(task.predict(X)[np.newaxis, ...])
        preds.append(
            forward_model(X, training=False).numpy()[np.newaxis, ...]
        )
    all_X = all_X[1:, ...]

    # Evaluate the designs using the oracle and the forward model.
    preds = np.concatenate(preds, axis=0)
    scores = np.concatenate(scores, axis=0)

    # Save the optimization results.
    np.save(os.path.join(logging_dir, "solution.npy"), all_X)
    np.save(os.path.join(logging_dir, "predictions.npy"), preds)
    np.save(os.path.join(logging_dir, "scores.npy"), scores)


def main():
    seed_everything(seed=pop_seed())
    if get_task_name() == os.environ["WARFARIN_TASK"]:
        coms_partial()
    else:
        coms_cleaned()


if __name__ == "__main__":
    main()
