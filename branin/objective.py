"""
Defines the oracle and surrogate objectives for the Branin toy task.

Author(s):
    Michael Yao

Citation(s):
    [1] Krishnamoorthy S, Mashkaria SM, Grover A. Generative pretraining
        for black-box optimization. Proc ICML. (2023).
        https://doi.org/10.48550/arXiv.2206.10786
    [2] Krishnamoorthy S, Mashkaria SM, Grover A. Diffusion models for
        black-box optimization. Proc ICML. (2023).
        https://doi.org/10.48550/arXiv.2306.07180

Licensed under the MIT License. Copyright University of Pennsylvania 2023.
"""
from __future__ import annotations
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pickle
import sys
import torch
import torch.nn as nn
from scipy.stats import qmc
from sklearn.neural_network import MLPRegressor
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

sys.path.append(".")
from experiment.utility import seed_everything, plot_config


class Oracle:
    """
    Default 2D Braining function parameters are as in the cited works above.
    """

    def __init__(
        self,
        a: float = 1.0,
        b: float = 5.1 / (4.0 * np.pi * np.pi),
        c: float = 5.0 / np.pi,
        r: float = 6.0,
        s: float = 10.0,
        t: float = 1.0 / (8.0 * np.pi),
        x1_range: Tuple[float] = (-5.0, 10.0),
        x2_range: Tuple[float] = (0.0, 15.0)
    ):
        """
        Args:
            a: a parameter for the negative of the 2D Brainin function.
            b: b parameter for the negative of the 2D Brainin function.
            c: c parameter for the negative of the 2D Brainin function.
            r: r parameter for the negative of the 2D Brainin function.
            s: s parameter for the negative of the 2D Brainin function.
            t: t parameter for the negative of the 2D Brainin function.
            x1_range: range of valid inputs for the argument x1.
            x2_range: range of valid inputs for the argument x2.
        """
        self.a, self.b, self.c, self.r, self.s, self.t = a, b, c, r, s, t
        self.x1_range, self.x2_range = x1_range, x2_range

    def __call__(
        self, x1: Union[float, np.ndarray], x2: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        """
        Computes the negative of the 2D Brainin function at (x1, x2).
        Input:
            x1: a value or batch of values for the x1 coordinate.
            x2: a value or batch of values for the x2 coordinate.
        Returns:
            -f_br(x1, x2).
        """
        return -(self.s * (1.0 - self.t) * np.cos(x1)) - self.s - (
            self.a * np.power(
                x2 - (self.b * x1 * x1) + (self.c * x1) - self.r, 2
            )
        )

    def _validate_inputs(
        self, x1: Union[float, np.ndarray], x2: Union[float, np.ndarray]
    ) -> bool:
        """
        Validates the inputs x1 and x2.
        Input:
            x1: a value or batch of values for the x1 coordinate.
            x2: a value or batch of values for the x2 coordinate.
        Returns:
            Whether x1 and x2 lie in the specified region of interest.
        """
        if isinstance(x1, float):
            x1 = np.array([x1])
        if isinstance(x2, float):
            x2 = np.array([x2])
        if not np.all(x1 <= max(self.x1_range)) or (
            not np.all(x1 >= min(self.x1_range))
        ):
            return False
        return np.all(x2 <= max(self.x2_range)) and (
            np.all(x2 >= min(self.x2_range))
        )

    @property
    def optima(self) -> Sequence[Tuple[float]]:
        """
        Returns the optimal points according to the oracle.
        Input:
            None.
        Returns:
            A list of (x1, x2) pairs corresponding to the optimal points.
        """
        return [(-np.pi, 12.275), (np.pi, 2.275), (9.42478, 2.475)]

    def show(self, savepath: Optional[Union[Path, str]] = None) -> None:
        """
        Generates a contour plot of the objective function.
        Input:
            savepath: optional path to save the plot to.
        Returns:
            None.
        """
        plot_config(fontsize=20)
        x1 = np.linspace(min(self.x1_range), max(self.x1_range), num=1000)
        x2 = np.linspace(min(self.x2_range), max(self.x2_range), num=1000)
        x1, x2 = np.meshgrid(x1, x2)
        plt.figure()
        plt.contour(x1, x2, self(x1, x2), levels=100, cmap="jet")
        plt.xlabel(r"$x_1$")
        plt.ylabel(r"$x_2$")
        plt.colorbar()
        plt.scatter(
            [x1 for x1, _ in self.optima],
            [x2 for _, x2 in self.optima],
            color="k",
            marker=(5, 1),
            label="Maxima"
        )
        plt.legend(loc="lower right")
        if savepath is None:
            plt.show()
        else:
            plt.savefig(
                savepath, dpi=600, transparent=True, bbox_inches="tight"
            )
        plt.close()


class Surrogate:
    def __init__(
        self,
        seed: int = 42,
        x1_range: Tuple[float] = (-5.0, 10.0),
        x2_range: Tuple[float] = (0.0, 15.0)
    ):
        """
        Args:
            seed: random seed. Default 42.
            x1_range: range of valid inputs for the argument x1.
            x2_range: range of valid inputs for the argument x2.
        """
        self.seed = seed
        self.x1_range, self.x2_range = x1_range, x2_range
        self.sampler = qmc.Sobol(d=2, scramble=True, seed=self.seed)
        self.model = MLPRegressor(
            hidden_layer_sizes=((2048,) * 2),
            learning_rate_init=0.0002,
            activation="relu"
        )

    def _generate_dataset(
        self, oracle: Oracle, n: int = 1000, exclude_top_p: float = 0.2
    ) -> Tuple[np.ndarray]:
        """
        Generates a dataset observed from nature.
        Input:
            oracle: oracle function to compare the surrogate against.
            n: number of datums to generate. Default 5000.
            exclude_top_p: fraction of highest scoring datums to exclude
                from the dataset. Default not performed.
        Returns:
            X: an Nx2 array of N (x1, x2) observations.
            y: an N vector of N corresponding function values.
        """
        X = self.sampler.random_base2(
            m=int(np.log2(1 << (n - 1).bit_length()))
        )
        X = qmc.scale(
            X[:n],
            [min(self.x1_range), min(self.x2_range)],
            [max(self.x1_range), max(self.x2_range)]
        )
        y = oracle(X[:, 0], X[:, -1])
        idxs = np.argsort(y)
        if exclude_top_p > 0.0:
            idxs = idxs[:-round(exclude_top_p * n)]
        return X[idxs], y[idxs]

    def __call__(self, X: np.ndarray) -> np.ndarray:
        """
        Forward pass through the trained model.
        Input:
            X: an Nx2 array of N (x1, x2) observations.
        Returns:
            ypred: an N vector of N predicted function values.
        """
        return self.y_min + ((self.y_max - self.y_min) * self.model.predict(X))

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Trains the model given in the input set of observations.
        Input:
            X: an Nx2 array of N (x1, x2) observations.
            y: an N vector of N corresponding function values.
        Returns:
            None.
        """
        self.y_min, self.y_max = np.min(y), np.max(y)
        self.model.fit(X, (y - self.y_min) / (self.y_max - self.y_min))
        self.train_X = X
        return

    def save_model(self, savepath: Union[Path, str]) -> None:
        """
        Saves the trained model to a specified path.
        Input:
            savepath: filepath to save the model to.
        Returns:
            None.
        """
        checkpoint = {
            "model": self.model,
            "train_X": self.train_X,
            "y_min": self.y_min,
            "y_max": self.y_max,
            "seed": self.seed,
            "x1_range": self.x1_range,
            "x2_range": self.x2_range
        }
        with open(savepath, "wb") as f:
            pickle.dump(checkpoint, f)
        return

    @staticmethod
    def load_from_checkpoint(ckpt: Union[Path, str]) -> Surrogate:
        """
        Loads a pretrained model from a specified path.
        Input:
            ckpt: file path to load the model from.
        Returns:
            The loaded pretrained Surrogate model.
        """
        with open(ckpt, "rb") as f:
            checkpoint = pickle.load(f)
        model = Surrogate(
            seed=checkpoint["seed"],
            x1_range=checkpoint["x1_range"],
            x2_range=checkpoint["x2_range"]
        )
        model.model = checkpoint["model"]
        model.y_min, model.y_max = checkpoint["y_min"], checkpoint["y_max"]
        model.train_X = checkpoint["train_X"]
        return model

    def test(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        do_plot: bool = False,
        savepath: Optional[Union[Path, str]] = None,
    ) -> float:
        """
        Evaluates the trained model performance.
        Input:
            X: test dataset of datums to evaluate on.
            y: test dataset of ground truth oracle values to evaluate on.
            do_plot: whether to generate the histogram of test results.
            savepath: optional path to save the performance plot to.
        Returns:
            The RMSE regression error on the test dataset.
        """
        rmse = np.sqrt(np.mean(np.square(y - self(X))))
        if not do_plot:
            return rmse
        plt.figure(figsize=(10, 5))
        plt.hist(self(X) - y, bins=100, color="k", alpha=0.5)
        plt.xlabel("Surrogate - Oracle Branin Function Value", fontsize=12)
        plt.ylabel("Count", fontsize=12)
        plt.annotate(
            f"RMSE = {rmse:.3f}",
            (0.825, 0.925),
            xycoords="axes fraction",
            fontsize=12
        )
        if savepath is None:
            plt.show()
        else:
            plt.savefig(
                savepath, dpi=600, transparent=True, bbox_inches="tight"
            )
        plt.close()
        return rmse


class TorchSurrogate(nn.Module):
    def __init__(self, model: Union[Path, str]):
        """
        Args:
            model: file path to load the model from.
        """
        super().__init__()
        model, module = Surrogate.load_from_checkpoint(model), []
        self.y_min, self.y_max = model.y_min, model.y_max
        self.ref_dataset = model.train_X
        for i, (W, b) in enumerate(
            zip(model.model.coefs_, model.model.intercepts_)
        ):
            module.append(nn.Linear(*W.shape))
            with torch.no_grad():
                module[-1].weight.copy_(
                    torch.tensor(W.T).to(module[-1].weight)
                )
                module[-1].bias.copy_(torch.tensor(b).to(module[-1].bias))
            if i < len(model.model.coefs_) - 1:
                module.append(nn.ReLU())
        self.model = nn.Sequential(*module)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the surrogate model.
        Input:
            X: an Nx2 array of N (x1, x2) points.
        Returns:
            The surrogate model predictions.
        """
        return self.y_min + ((self.y_max - self.y_min) * self.model(X))


def from_sklearn_model(model: Union[Path, str]) -> nn.Module:
    """
    Loads a trained scikit-learn MLPRegressor model as a PyTorch model.
    Input:
        model: file path to load the model from.
    Returns:
        The trained model loaded as a PyTorch module.
    """
    return TorchSurrogate(model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Surrogate Branin Objective Trainer"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed. Default 42."
    )
    seed = parser.parse_args().seed

    # Define the oracle and surrogate objectives.
    seed_everything(seed)
    oracle = Oracle()
    surrogate = Surrogate()

    # Generate the dataset.
    X, y = surrogate._generate_dataset(oracle)
    idxs = np.arange(len(X))
    np.random.shuffle(idxs)
    X, y = X[idxs], y[idxs]

    # Create the training and test dataset partitions.
    test_frac = 0.1
    test_len = -round(test_frac * len(X))
    X_train, y_train = X[:-test_len], y[:-test_len]
    X_test, y_test = X[-test_len:], y[-test_len:]

    # Train and evaluate the model.
    surrogate.fit(X_train, y_train)
    surrogate.save_model(f"./branin/ckpts/surrogate_{seed}.pkl")
    print(f"RMSE (seed {seed}): {surrogate.test(X_test, y_test):.3f}")
