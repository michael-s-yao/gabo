"""
Univariate parameter annealing schedule implementation.

Author(s):
    Michael Yao @michael-s-yao

Licensed under the MIT License. Copyright University of Pennsylvania 2023.
"""
from abc import ABC
import numpy as np
from typing import Optional, Sequence


class AnnealingSchedule(ABC):
    """Base class implementation for a general annealing schedule."""

    def __init__(self, start: float, end: float, T_warmup: int):
        """
        Args:
            start: starting value for annealing schedule.
            end: ending value for annealing schedule.
            T_warmup: time step at which to start the annealing schedule.
        """
        self.start, self.end, self.counter = start, end, 0
        self.T_warmup = T_warmup
        assert self.start > self.end

    def __call__(self) -> float:
        """
        Returns the current parameter value based on the specified annealing
        schedule.
        Input:
            None.
        Returns:
            Current parameter value based on the specified annealing schedule.
        """
        raise NotImplementedError

    def step(self) -> None:
        """
        Takes a step forward in the annealing schedule.
        Input:
            None.
        Returns:
            None.
        """
        self.counter += 1
        return


class ConstantSchedule(AnnealingSchedule):
    """Constant annealing schedule for a univariate parameter."""

    def __init__(self, value: float):
        """
        Args:
            value: constant value for the univariate parameter.
        """
        super().__init__(value, -np.inf, 0)

    def __call__(self) -> float:
        return self.start


class SquareRootSchedule(AnnealingSchedule):
    """Square root annealing schedule for a univariate parameter."""

    def __init__(
        self, start: float = 1.0, end: float = 0.0, T_warmup: Optional[int] = 0
    ):
        """
        Args:
            start: starting value for annealing schedule.
            end: ending value for annealing schedule.
            T_warmup: time step at which to start the annealing schedule.
        """
        super().__init__(start, end, T_warmup)
        self.eps = np.finfo(np.float32).eps

    def __call__(self) -> float:
        if self.counter <= self.T_warmup:
            return self.start
        return self.end + (
            (self.start - self.end) / (
                np.sqrt(1 + (self.counter - self.T_warmup)) + self.eps
            )
        )


class FactorSchedule(AnnealingSchedule):
    """Factor annealing schedule for a univariate parameter."""

    def __init__(
        self,
        start: float = 1.0,
        end: float = 0.0,
        gamma: float = 0.9,
        milestones: Optional[Sequence[int]] = None,
        T_warmup: Optional[int] = 0
    ):
        """
        Args:
            start: starting value for annealing schedule.
            end: ending value for annealing schedule.
            gamma: multiplicative factor to decay the parameter by.
            milestones: optional parameter to specify at which time steps the
                parameter should decay by gamma. If not specified, then the
                parameter will decay after every step.
            T_warmup: time step at which to start the annealing schedule.
        """
        super().__init__(start, end, T_warmup)
        self.factor = 1.0
        self.gamma = min(max(gamma, 0.0), 1.0)
        self.milestones = milestones
        if self.milestones is not None:
            assert min(self.milestones) > self.T_warmup

    def __call__(self) -> float:
        if self.counter <= self.T_warmup:
            return self.start
        return self.end + ((self.start - self.end) * self.factor)

    def step(self) -> None:
        super().step()
        if self.counter <= self.T_warmup:
            return
        elif self.milestones is None or self.counter in self.milestones:
            self.factor = self.factor * self.gamma


class CosineSchedule(AnnealingSchedule):
    """Cosine annealing schedule for a univariate parameter."""

    def __init__(
        self,
        start: float = 1.0,
        end: float = 0.0,
        T: int = 50,
        T_warmup: Optional[int] = 0
    ):
        """
        Args:
            start: starting value for annealing schedule.
            end: ending value for annealing schedule.
            T: time step at which to reach the final endpoint of the schedule.
            T_warmup: time step at which to start the annealing schedule.
        """
        super().__init__(start, end, T_warmup)
        self.T = T
        assert self.T > self.T_warmup

    def __call__(self) -> float:
        if self.counter <= self.T_warmup:
            return self.start
        if self.counter > self.T:
            return self.end
        return self.end + (
            0.5 * (self.start - self.end) * (
                1.0 + np.cos(
                    np.pi * (self.counter - self.T_warmup) / (
                        self.T - self.T_warmup
                    )
                )
            )
        )


def build_schedule(alpha: str) -> AnnealingSchedule:
    """
    Builds an annealing schedule based on the alpha argument provided.
    Input:
        alpha: either a float between 0.0 and 1.0 inclusive or a string
            specifying an annealing schedule.
    Returns:
        The specified annealing schedule.
    """
    if alpha.replace(".", "", 1).isdigit():
        return ConstantSchedule(float(alpha))
    elif alpha == "SquareRootSchedule":
        return SquareRootSchedule(T_warmup=25)
    elif alpha == "FactorSchedule":
        return FactorSchedule(T_warmup=25)
    elif alpha == "CosineSchedule":
        return CosineSchedule(T_warmup=0, T=50)
    raise NotImplementedError(f"Unrecognized annealing schedule {alpha}.")
