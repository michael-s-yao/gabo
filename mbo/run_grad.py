#!/usr/bin/env python3
"""
Main driver program for baseline gradient ascent MBO methods.

Author(s):
    Michael Yao @michael-s-yao

Adapted from the design-baselines GitHub repo from @brandontrabucco.
https://github.com/brandontrabucco/design-baselines/design_baselines/
gradient_ascent/__init__.py

Licensed under the MIT License. Copyright University of Pennsylvania 2023.
"""
import argparse
import numpy as np
import os
import sys
import tensorflow as tf
from pathlib import Path
from typing import Optional, Sequence, Union

sys.path.append(".")
import mbo
import design_bench
from design_baselines.data import StaticGraphTask, build_pipeline
from design_baselines.gradient_ascent.trainers import MaximumLikelihood
from design_baselines.gradient_ascent.nets import ForwardModel
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
        "--aggregation-method",
        type=str,
        default="None",
        choices=("None", "mean", "min"),
        help="Aggregation method. Default None."
    )
    parser.add_argument(
        "--particle-evaluate-gradient-steps",
        type=int,
        default=32,
        help="Number of gradient ascent steps to perform per sample."
    )
    parser.add_argument(
        "--evaluation-samples",
        type=int,
        default=8,
        help="Number of datums to perform gradient ascent from."
    )

    return parser.parse_args()


def reduce_preds(
    predictions: Sequence[tf.Tensor], aggregation_method: str
) -> Optional[tf.Tensor]:
    """
    Reduces an ensemble of predictions from different models into a single
    prediction.
    Input:
        predictions: a sequence of predictions from an ensemble of models.
        aggregation_method: aggregation method. One of [None, `mean`, `min`].
    Returns:
        A tensor containing the final prediction from the ensemble of models.
    """
    if aggregation_method is None or aggregation_method == "None":
        return predictions[0]
    elif aggregation_method == "mean":
        return tf.reduce_mean(predictions, axis=0)
    elif aggregation_method == "min":
        return tf.reduce_min(predictions, axis=0)
    return None


def gradient_ascent(
    task_name: str,
    logging_dir: Union[Path, str],
    aggregation_method: Optional[str] = None,
    num_epochs: int = 100,
    batch_size: int = 128,
    val_size: int = 200,
    forward_model_lr: float = 0.0003,
    solver_steps: int = 32,
    solver_lr: float = 0.01,
    solver_samples: int = 8
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
            surrogate model. Default 32.
        solver_lr: step size for gradient ascent against the trained surrogate
            model. Default 0.01.
        solver_samples: total number of final designs to return. Default 8.
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
    num_models = 5
    if aggregation_method is None or aggregation_method == "None":
        num_models = 1
    forward_models = [
        ForwardModel(
            task.input_shape,
            activations=("leaky_relu", "leaky_relu"),
            hidden_size=2048
        )
        for _ in range(num_models)
    ]

    # Scale the learning rate based on the number of design dimensions.
    if hasattr(task.wrapped_task.dataset, "grad_mask"):
        solver_lr *= np.sqrt(np.sum(task.wrapped_task.dataset.grad_mask))
    else:
        solver_lr *= np.sqrt(np.prod(task.input_shape))

    trs = []
    for i, fm in enumerate(forward_models):
        train, validate = build_pipeline(
            x=task.x,
            y=task.y,
            batch_size=batch_size,
            val_size=val_size,
            bootstraps=1
        )
        trainer = MaximumLikelihood(
            fm,
            forward_model_optim=tf.keras.optimizers.Adam,
            forward_model_lr=forward_model_lr,
            noise_std=0.0
        )
        trs.append(trainer)
        trainer.launch(train, validate, logger=logger, epochs=num_epochs)

    # Select the top k initial designs from the dataset as starting points.
    if task_name in mbo.CONDITIONAL_TASKS:
        X = tf.concat([x for x, _ in validate], axis=0)
        solver_steps = solver_steps * solver_samples
    else:
        indices = tf.math.top_k(np.squeeze(task.y, axis=-1), k=solver_samples)
        indices = indices[1]
        X = tf.gather(task.x, indices, axis=0)
    all_X = X.numpy()[np.newaxis, ...]

    # Perform gradient ascent on the score through the surrogate model.
    for _ in range(solver_steps):
        with tf.GradientTape() as tape:
            tape.watch(X)
            predictions = [
                fm.get_distribution(X).mean() for fm in forward_models
            ]
            score = reduce_preds(predictions, aggregation_method)
        grads = tape.gradient(score, X)

        # Use the conservative optimizer to update the solution.
        if hasattr(task.wrapped_task.dataset, "grad_mask"):
            grads = grads * tf.convert_to_tensor(
                task.wrapped_task.dataset.grad_mask
            )
        X = X + (solver_lr * grads)
        all_X = np.concatenate([all_X, X.numpy()[np.newaxis, ...]], axis=0)
    all_X = all_X[1:, ...]

    # Evaluate the designs using the oracle and the forward model.
    preds = reduce_preds(
        [
            fm.get_distribution(all_X.reshape(-1, *task.input_shape)).mean()
            for fm in forward_models
        ],
        aggregation_method
    )
    preds = preds.numpy().reshape(all_X.shape[0], all_X.shape[1], 1)
    scores = task.predict(all_X.reshape(-1, *task.input_shape))
    scores = scores.numpy().reshape(all_X.shape[0], all_X.shape[1], 1)
    if task.is_discrete:
        all_X = task.to_integers(all_X)

    # Save the optimization results.
    np.save(os.path.join(logging_dir, "solution.npy"), all_X)
    np.save(os.path.join(logging_dir, "predictions.npy"), preds)
    np.save(os.path.join(logging_dir, "scores.npy"), scores)


def main():
    args = vars(build_args())
    seed_everything(seed=args.pop("seed"))
    args["task_name"] = args.pop("task")
    args["solver_steps"] = args.pop("particle_evaluate_gradient_steps")
    args["solver_samples"] = args.pop("evaluation_samples")
    gradient_ascent(**args)


if __name__ == "__main__":
    main()
