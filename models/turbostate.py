"""
TurboState implementation to track the state of the trust region.

Author(s):
    Yimeng Zeng @yimeng-zeng
    Michael Yao @michael-s-yao

Licensed under the MIT License. Copyright University of Pennsylvania 2023.
"""
import math
import torch


class TurboState:
    """Tracks the state of the trust region for Bayesian optimization."""

    def __init__(
        self,
        length: float = 1.0,
        length_min: float = 0.5 ** 7,
        length_max: float = 10.0,
        failure_patience: int = 5,
        success_patience: int = 10,
    ):
        """
        Args:
            length: initial size of the trust region.
            length_min: minimum valid size of the trust region under which
                a restart is triggered.
            length_max: maximum valid size of the trust region.
            failure_patience: patience before trust region is shrunk.
            success_patience: patience before trust region is expanded.
        """
        self.length = length
        self.length_min, self.length_max = length_min, length_max
        self.failure_counter, self.success_counter = 0, 0
        self.failure_patience = 5
        self.success_patience = 10
        self.rel_improvement = 1e-3
        self.best_value = -float("inf")
        self.restart_triggered = False

    def update(self, y: torch.Tensor) -> None:
        """
        Updates the state internal variables given objective values y.
        Input:
            y: input objective values.
        Returns:
            None.
        """
        if torch.max(y) > self.best_value + (
            self.rel_improvement * math.fabs(self.best_value)
        ):
            self.success_counter += 1
            self.failure_counter = 0
        else:
            self.success_counter = 0
            self.failure_counter += 1

        # Expand trust region.
        if self.success_counter >= self.success_patience:
            self.length = min(2.0 * self.length, self.length_max)
            self.success_counter = 0
        # Shrink trust region.
        elif self.failure_counter >= self.failure_patience:
            self.length /= 2.0
            self.failure_counter = 0

        self.best_value = max(self.best_value, torch.max(y).item())
        if self.length < self.length_min:
            self.restart_triggered = True
        return
