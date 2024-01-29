"""
Implements a conditional Conservative Objective Model (COM) where only a subset
of the input design dimensions are optimized over and the rest are treated as
frozen "conditions".

Author(s):
    Michael Yao @michael-s-yao

Citation(s):
    [1] Trabucco B, Kumar A, Geng X, Levine S. Conservative objective models
        for effective offline model-based optimization. Proc ICML 139:10358-
        68. (2021). http://proceedings.mlr.press/v139/trabucco21a.html

Adapted from the design-baselines GitHub repo from @brandontrabucco at
https://github.com/brandontrabucco/design-baselines/design_baselines/
coms_cleaned/trainers.py

Licensed under the MIT License. Copyright University of Pennsylvania 2024.
"""
import numpy as np
import sys
import tensorflow as tf
from tensorflow.keras.optimizers import Optimizer, Adam
from typing import Tuple

sys.path.append(".")
from design_baselines.coms_cleaned.trainers import ConservativeObjectiveModel


class ConditionalConservativeObjectiveModel(ConservativeObjectiveModel):
    def __init__(
        self,
        grad_mask: np.ndarray,
        forward_model: tf.keras.Model,
        forward_model_opt: Optimizer = Adam,
        forward_model_lr: float = 0.001,
        alpha: float = 1.0,
        alpha_opt: Optimizer = Adam,
        alpha_lr: float = 0.01,
        overestimation_limit: float = 0.5,
        particle_lr: float = 0.05,
        particle_gradient_steps: float = 50,
        entropy_coefficient: float = 0.9,
        noise_std: float = 0.0
    ):
        """
        Args:
            grad_mask: a mask of input design dimensions that can be optimized
                over. The mask should be True for dimensions that will be
                optimized over and False for frozen condition dimensions.
            forward_model: the surrogate objective model.
            forward_model_opt: the optimizer for the forward model.
            forward_model_lr: the learning rate for the forward model.
            alpha: the initial value of the Lagrange multiplier in the
                conservatism objective of the forward model.
            alpha_opt: the optimizer for the Lagrange multiplier.
            alpha_lr: the learning rate for the Lagrange multiplier.
            overestimation_limit: the degree to which the predictions of
                the model overestimate the true score function.
            particle_lr: the learning rate for the gradient ascent
                optimizer used to find adversarial solution particles.
            particle_gradient_steps: number of gradient ascent steps used to
                find adversarial solution particles.
            entropy_coefficient: the entropy bonus added to the loss function
                when updating solution particles with gradient ascent.
            noise_std: the standard deviation of the gaussian noise added to
                designs when training the forward model.
        """
        super(ConditionalConservativeObjectiveModel, self).__init__(
            forward_model=forward_model,
            forward_model_opt=forward_model_opt,
            forward_model_lr=forward_model_lr,
            alpha=alpha,
            alpha_opt=alpha_opt,
            alpha_lr=alpha_lr,
            overestimation_limit=overestimation_limit,
            particle_lr=particle_lr,
            particle_gradient_steps=particle_gradient_steps,
            entropy_coefficient=entropy_coefficient,
            noise_std=noise_std
        )
        self.grad_mask = tf.convert_to_tensor(grad_mask)

    @tf.function(experimental_relax_shapes=True)
    def optimize(self, x: tf.Tensor, steps: int, **kwargs) -> tf.Tensor:
        """
        Finds adversarial input designs that using gradient ascent that
        maximize the conservatism of the model.
        Input:
            x: the starting point(s) in the design space for gradient ascent.
            steps: the number of gradient ascent steps to take.
        Returns:
            A new design or set of designs.
        """
        def partial_gradient_step(xt: tf.Tensor) -> Tuple[tf.Tensor]:
            """
            Implements a single step of partial gradient ascent.
            Input:
                xt: the current point(s) in the design space.
            Returns:
                The next iterative point(s) in the design space after a single
                step of gradient ascent.
            """
            with tf.GradientTape() as tape:
                tape.watch(xt)

                # Shuffle the designs for calculating entropy.
                shuffled_xt = tf.gather(
                    xt, tf.random.shuffle(tf.range(tf.shape(xt)[0]))
                )

                # Calculate the entropy using the gaussian kernel.
                entropy = tf.reduce_mean((xt - shuffled_xt) ** 2)

                # Calculate the predicted score according to the forward model.
                score = self.forward_model(xt, **kwargs)

                # Calculate the conservatism of the current set of particles.
                loss = self.entropy_coefficient * entropy + score

            return tf.stop_gradient(
                xt + (
                    self.particle_lr * self.grad_mask * tape.gradient(loss, xt)
                )
            ),

        return tf.while_loop(
            lambda xt: True,
            partial_gradient_step,
            (x,),
            maximum_iterations=steps
        )[0]
