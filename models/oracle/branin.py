"""
Defines the oracle objective for the Branin toy task.

Author(s):
    Michael Yao

Citation(s):
    [1] Branin FH. Widely convergent method for finding multiple solutions of
        simultaneous nonlinear equations. IBM J Res and Dev 16(5):504-22.
        (1972). https://doi.org/10.1147/rd.165.0504

Licensed under the MIT License. Copyright University of Pennsylvania 2023.
"""
import matplotlib.pyplot as plt
import numpy as np
import sys
import torch
import torch.nn as nn
from botorch.test_functions.synthetic import Branin
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

sys.path.append(".")
from utils import plot_config


class BraninOracle(nn.Module):
    def __init__(self, negate: bool = True):
        """
        Args:
            negate: whether to return the negative of the Branin function.
        """
        super().__init__()
        self.negate = negate
        self.oracle = Branin(negate=self.negate)

    def forward(
        self, X: Union[np.ndarray, torch.Tensor],
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Computes the negative of the 2D Brainin function at a specified point.
        Input:
            X: a coordinate or batch of coordinates.
        Returns:
            The function value(s) at X.
        """
        X = X[np.newaxis] if X.ndim == 1 else X
        is_np = isinstance(X, np.ndarray)
        X = torch.from_numpy(X) if is_np else X
        y = self.oracle(X)
        return y.detach().cpu().numpy() if is_np else y

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

    def show(
        self,
        x1_range: Tuple[float] = (-5.0, 10.0),
        x2_range: Tuple[float] = (0.0, 15.0),
        savepath: Optional[Union[Path, str]] = None
    ) -> None:
        """
        Generates a contour plot of the objective function.
        Input:
            x1_range: range of inputs for the argument x1 to plot.
            x2_range: range of inputs for the argument x2 to plot.
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
