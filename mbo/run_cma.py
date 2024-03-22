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
import tensorflow as tf
from pathlib import Path
from typing import Optional, Union

sys.path.append(".")
import mbo
import design_bench
from design_baselines.data import StaticGraphTask, build_pipeline
from design_baselines.cma_es.trainers import Ensemble
from design_baselines.cma_es.nets import ForwardModel
from models.logger import DummyLogger
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
        "--particle-evaluate-gradient-steps",
        type=int,
        default=128,
        help="Number of gradient ascent steps to perform per sample."
    )
    parser.add_argument(
        "--evaluation-samples",
        type=int,
        default=16,
        help="Number of datums to perform gradient ascent from."
    )

    return parser.parse_args()


def cma_es(
    task_name: str,
    logging_dir: Optional[Union[Path, str]] = None,
    num_epochs: int = 100,
    batch_size: int = 128,
    val_size: int = 200,
    forward_model_lr: float = 0.0003,
    solver_steps: int = 128,
    solver_lr: float = 0.01,
    solver_samples: int = 16,
    cma_sigma: float = 0.5,
    seed: int = 42
) -> None:
    """
    Train a surrogate objective model and perform gradient ascent to solve a
    Model-Based Optimization (MBO) task. This method is adapted directly from
    the gradient_ascent() baseline method from the design-baselines repo cited
    at the top of this source code file.
    Input:
        task_name: name of the offline model-based optimization (MBO) task.
        logging_dir: directory to log the optimization results to.
        aggregation_method: optional specification of aggregating the results
            of an ensemble of surrogate models. One of [None, `mean`, `min`].
        num_epochs: number of epochs for the surrogate model. Default 100.
        batch_size: the number of samples to load in every batch when drawing
            samples from the training and validation sets. Default 128.
        val_size: the number of samples randomly chosen to be in the validation
            set. Default 200.
        forward_model_lr: learning rate for training the surrogate model.
            Default 3e-4.
        solver_steps: number of steps for gradient ascent against the trained
            surrogate model. Default 128.
        solver_lr: step size for gradient ascent against the trained surrogate
            model. Default 0.01.
        solver_samples: total number of final designs to return. Default 16.
        cma_sigma: TODO. Default 0.5.
        seed: random seed. Default 42.
    Returns:
        None.
    """
    logger = DummyLogger(logging_dir)
    task = StaticGraphTask(task_name, relabel=False)

    # Task-specific setup.
    if task.is_discrete:
        task.map_to_logits()
    if task_name == os.environ["CHEMBL_TASK"]:
        task.map_normalize_y()

    # Make several keras neural networks with different architectures.
    num_bootstraps = 5
    forward_models = [
        ForwardModel(task.input_shape, num_layers=2, hidden_size=2048)
        for _ in range(num_bootstraps)
    ]

    # Scale the learning rate based on the number of design dimensions.
    if hasattr(task.wrapped_task.dataset, "grad_mask"):
        solver_lr *= np.sqrt(np.sum(task.wrapped_task.dataset.grad_mask))
    else:
        solver_lr *= np.sqrt(np.prod(task.input_shape))

    ensemble = Ensemble(
        forward_models,
        forward_model_optim=tf.keras.optimizers.Adam,
        forward_model_lr=forward_model_lr
    )
    train, validate = build_pipeline(
        x=task.x,
        y=task.y,
        batch_size=batch_size,
        val_size=val_size,
        bootstraps=num_bootstraps
    )
    ensemble.launch(train, validate, logger=logger, epochs=num_epochs)

    # Select the top k initial designs from the dataset as starting points.
    if task_name in mbo.CONDITIONAL_TASKS:
        X = tf.concat([x for x, _ in validate], axis=0)
        solver_steps = solver_steps * solver_samples
    else:
        indices = tf.math.top_k(np.squeeze(task.y, axis=-1), k=solver_samples)
        indices = indices[1]
        X = tf.gather(task.x, indices, axis=0)

    # Create a fitness function for optimizing the expected task score.
    def fitness(_x: tf.Tensor) -> float:
        """
        Computs the fitness of an input design.
        Inputs:
            _x: an input 1-D design vector.
        Returns:
            The estimated fitness for optimizing the expected task score.
        """
        _x = tf.reshape(_x, task.x.shape[1:])[tf.newaxis]
        value = ensemble.get_distribution(_x).mean()
        return (-value[0].numpy()).tolist()[0]

    all_X = []
    for i in range(solver_samples):
        xi = X[i].numpy().flatten().tolist()
        es = cma.CMAEvolutionStrategy(xi, cma_sigma, {"seed": seed})
        step = 0
        result = []
        while not es.stop() and step < solver_steps:
            solutions = es.ask()
            es.tell(solutions, [fitness(x) for x in solutions])
            result.append(es.result.xbest)
            step += 1
        result = ([xi] * (solver_steps - len(result))) + result
        all_X.append(tf.stack(result, axis=0))
    all_X = tf.stack(all_X, axis=0)

    if task.is_discrete:
        preds = np.array([
            ensemble.get_distribution(_x[tf.newaxis]).mean()
            for _x in tf.reshape(all_X, (-1, *X.shape[1:]))
        ])
    else:
        preds = np.array([
            ensemble.get_distribution(_x).mean() for _x in all_X
        ])
    preds = preds.reshape(all_X.shape[0], all_X.shape[1], 1)
    scores = task.predict(
        all_X.numpy().reshape(-1, *task.input_shape).astype(np.float32)
    )
    scores = scores.numpy().reshape(all_X.shape[0], all_X.shape[1], 1)
    if task.is_discrete:
        all_X = task.to_integers(all_X)

    # Save the optimization results.
    if logging_dir is not None:
        os.makedirs(logging_dir, exist_ok=True)
        np.save(os.path.join(logging_dir, "solution.npy"), all_X.numpy())
        np.save(os.path.join(logging_dir, "predictions.npy"), preds)
        np.save(os.path.join(logging_dir, "scores.npy"), scores)


def main():
    args = vars(build_args())
    seed_everything(args["seed"])
    args["task_name"] = args.pop("task")
    args["solver_steps"] = args.pop("particle_evaluate_gradient_steps")
    args["solver_samples"] = args.pop("evaluation_samples")
    cma_es(**args)


if __name__ == "__main__":
    main()
