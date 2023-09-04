"""
In-distribution regularization functions.

Author(s):
    Michael Yao

Licensed under the MIT License. Copyright University of Pennsylvania 2023.
"""
import torch
import torch.nn as nn
from typing import Optional, Tuple, Union

from models.discriminator import Discriminator


class Regularization(nn.Module):
    """In-distribution regularization functions."""

    def __init__(
        self,
        method: Optional[str] = None,
        x_dim: Optional[Tuple[int]] = (1, 28, 28),
        p: Optional[Union[int, str]] = 1
    ):
        """
        Args:
            method: method of regularization. One of [`None`, `fid`,
                `gan_loss`, `importance_weighting`, `wasserstein_distance`].
            x_dim: dimensions CHW of the output image from the generator G.
                Default MNIST dimensions (1, 28, 28).
            p: Wasserstein distance norm. p >= 1 or p == `inf`. Must be
                provided if the method of Regularization is
                `_wasserstein_distance`.
        """
        super().__init__()
        self.method = method
        self.D = Discriminator(x_dim=x_dim)
        self.p = p

    def forward(
        self, xp: torch.Tensor, xq: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward propagation through a regularization function.
        Input:
            xp
        Returns:
            Generated sample x = G(z).
        """
        if self.method is None or self.method == "":
            return 0.0

        if self.method.lower() == "gan_loss":
            return torch.mean(self._gan_loss(xq))
        elif self.method.lower() == "importance_weighting":
            return torch.mean(self._importance_weight(xp)) - 1.0
        elif self.method.lower() == "wasserstein_distance":
            return torch.mean(self._wasserstein_distance(xp, xq))
        elif self.method.lower() == "fid":
            return self._fid(xp, xq)
        else:
            raise NotImplementedError(
                f"Regularization method {self.method} not implemented."
            )

    def _gan_loss(self, xq: torch.Tensor) -> torch.Tensor:
        """
        Computes the log of the source discriminator output on the generated
        samples `xq` from `q(x)`.
        Input:
            xq: samples from source distribution `q(x)`.
        Returns:
            -log(D(xq)).
        """
        return -1.0 * torch.log(self.D(xq))

    def _importance_weight(self, xp: torch.Tensor) -> torch.Tensor:
        """
        Computes the estimated importance weights w(x) = q(x) / p(x) given `xp`
        samples from `p(x)` and a source discriminator.
        Input:
            xp: samples from source distribution `p(x)`.
        Returns:
            Estimated importance weights at those values of x.
        """
        return (1.0 / self.D(xp)) - 1.0

    def _wasserstein_distance(
        self, xp: torch.Tensor, xq: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes the empirical p-Wasserstein distance between `xp` samples from
        `p(x)` and `xq` samples from `q(x)`.
        Input:
            xp: samples from source distribution `p(x)`.
            xq: samples from target distribution `q(x)`.
        Returns:
            Empirical p-Wasserstein distance between `p(x)` and `q(x)`.
        """
        # TODO: Implement Wasserstein distance.
        return 0

    def loss_D(self, xp: torch.Tensor, xq: torch.Tensor) -> torch.Tensor:
        """
        Source discriminator D loss.
        Input:
            xp: samples from source distribution `p(x)`.
            xq: samples from target distribution `q(x)`.
        Returns:
            Loss of source discriminator.
        """

        return torch.mean(
            -0.5 * (torch.log(self.D(xp)) + torch.log(1.0 - self.D(xq)))
        )
