"""
Implements the sampler policy to optimize patient warfarin dose.

Author(s):
    Michael Yao @michael-s-yao

Licensed under the MIT License. Copyright University of Pennsylvania 2023.
"""
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd


class Sampler(ABC):
    """Defines the sampler policy base class."""

    @abstractmethod
    def __call__(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Perturbs the warfarin dose of the table of patients and returns the
        the resulting perturbed sample.
        Input:
            X: a DataFrame of patent data.
        Returns:
            A new DataFrame of perturbed patient data.
        """
        raise NotImplementedError

    @abstractmethod
    def __len__(self) -> int:
        """
        Returns the number of times that the sampler has been called.
        Input:
            None.
        Returns:
            The number of times that the sampler has been called.
        """
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        """
        Resets the sampler.
        Input:
            None.
        Returns:
            None.
        """
        raise NotImplementedError

    @abstractmethod
    def feedback(self, cost: np.ndarray) -> None:
        """
        Keeps track of the best generated warfarin doses for each patient based
        on an input updated metric.
        Input:
            cost: predicted cost of the most recently generated warfarin dose.
        Returns:
            None.
        """
        raise NotImplementedError

    @abstractmethod
    def optimum(self) -> np.ndarray:
        """
        Returns the optimal warfarin dosing found for each patient based on
        the available sampling history.
        Input:
            None.
        Returns:
            A vector containing the optimal warfarin dosing for each patient.
        """
        raise NotImplementedError


class RandomSampler(Sampler):
    """Implements a random sampler policy."""

    def __init__(
        self,
        min_z_dose: float,
        max_z_dose: float,
        seed: int = 42,
        save_best_on_reset: bool = True
    ):
        """
        Args:
            min_z_dose: minimum warfarin dose (after normalization is applied).
            max_z_dose: maximum warfarin dose (after normalization is applied).
            seed: random seed. Default 42.
            save_best_on_reset: whether to save the best costs on reset.
        """
        self.min_z_dose, self.max_z_dose = min_z_dose, max_z_dose
        self.seed = seed
        self.save_best_on_reset = save_best_on_reset
        self.rng = np.random.RandomState(seed=self.seed)
        self.dose_key = "Therapeutic Dose of Warfarin"
        self.arg_best, self.best_cost, self.cache = None, None, []

    def __call__(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X[self.dose_key] = self._generate_doses(len(X))
        self.cache.append(X[self.dose_key])
        return X

    def __len__(self) -> int:
        return len(self.cache)

    def reset(self) -> None:
        if self.save_best_on_reset:
            self.best_cost, self.cache = self.optimum(), [self.optimum()]
            self.arg_best = np.zeros(len(self.arg_best), dtype=int)
            return
        self.arg_best, self.best_cost, self.cache = None, None, []

    def feedback(self, cost: np.ndarray) -> None:
        if self.arg_best is None or self.best_cost is None:
            self.best_cost = cost
            self.arg_best = np.zeros_like(cost, dtype=int)
            return
        self.arg_best = np.where(
            cost < self.best_cost, len(self) - 1, self.arg_best
        )

    def optimum(self) -> np.ndarray:
        opt, cache = np.zeros(len(self.arg_best)), np.array(self.cache)
        for i in range(len(self.arg_best)):
            opt[i] = cache[self.arg_best[i], i]
        return opt

    def _generate_doses(self, n: int) -> np.ndarray:
        """
        Generates a vector of n valid random doses.
        Input:
            n: number of random doses to generate.
        Returns:
            A vector of n valid random doses.
        """
        m, b = self.max_z_dose - self.min_z_dose, self.min_z_dose
        return (m * self.rng.rand(n)) + b
