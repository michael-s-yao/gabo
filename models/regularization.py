"""
In-distribution regularization functions.

Author(s):
    Michael Yao

Licensed under the MIT License. Copyright University of Pennsylvania 2023.
"""
import torch
import torch.nn as nn
from typing import Optional, Tuple

from models.critic import Critic, WeightClipper


class Regularization(nn.Module):
    """In-distribution regularization functions."""

    def __init__(
        self,
        method: Optional[str] = None,
        x_dim: Optional[Tuple[int]] = (1, 28, 28),
        c: Optional[float] = 0.01
    ):
        """
        Args:
            method: method of regularization. One of [`None`, `fid`,
                `gan_loss`, `importance_weighting`, `wasserstein`, `em`].
            x_dim: dimensions CHW of the output image from the generator G.
                Default MNIST dimensions (1, 28, 28).
            c: weight clipping to enforce 1-Lipschitz condition on source
                critic for `wasserstein`/`em` regularization algorithms.
        """
        super().__init__()
        self.method = method.lower() if method else method
        if self.method in ["gan_loss", "importance_weighting"]:
            self.D = Critic(x_dim=x_dim, use_sigmoid=True)
        elif self.method in ["wasserstein", "em"]:
            self.f = Critic(x_dim=x_dim, use_sigmoid=False)
            self.clipper = WeightClipper(c=c)
            self.clip()

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

        if self.method == "gan_loss":
            return torch.mean(self._gan_loss(xq))
        elif self.method == "importance_weighting":
            return torch.mean(torch.square(self._importance_weight(xp) - 1.0))
        elif self.method in ["wasserstein", "em"]:
            return self._wasserstein_distance_1(xp, xq)
        elif self.method == "fid":
            return self._fid(xp, xq)
        else:
            raise NotImplementedError(
                f"Regularization method {self.method} not implemented."
            )

    def _gan_loss(self, xq: torch.Tensor) -> torch.Tensor:
        """
        Computes the negative log of the source critic output on the generated
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
        samples from `p(x)` and a source critic.
        Input:
            xp: samples from source distribution `p(x)`.
        Returns:
            Estimated importance weights at those values of x.
        """
        return (1.0 / self.D(xp)) - 1.0

    def _wasserstein_distance_1(
        self, xp: torch.Tensor, xq: torch.Tensor
    ) -> torch.Tensor:
        """
        Estimates the 1-Wasserstein distance (i.e., Earth-Mover distance)
        between `p(x)` and `q(x)` using `xp` samples from `p(x)` and `xq`
        samples from `q(x)`.
        Input:
            xp: samples from source distribution `p(x)`.
            xq: samples from target distribution `q(x)`.
        Returns:
            Empirical 1-Wasserstein distance between `p(x)` and `q(x)`.
        """
        self.clip()
        return torch.mean(self.f(xp)) - torch.mean(self.f(xq))

    def critic_loss(self, xp: torch.Tensor, xq: torch.Tensor) -> torch.Tensor:
        """
        Source critic loss function implementation.
        Input:
            xp: samples from source distribution `p(x)`.
            xq: samples from target distribution `q(x)`.
        Returns:
            LogLoss of source critic D.
        """
        if self.method in ["gan_loss", "importance_weighting"]:
            return torch.mean(
                -0.5 * (torch.log(self.D(xp)) + torch.log(1.0 - self.D(xq)))
            )
        elif self.method in ["wasserstein", "em"]:
            return -1.0 * self._wasserstein_distance_1(xp, xq)

    def clip(self) -> torch.Tensor:
        """
        Clips the weights of self.f to [-self.clipper, self.clipper].
        Input:
            None.
        Returns:
            None.
        """
        self.f.apply(self.clipper)
