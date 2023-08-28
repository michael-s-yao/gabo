"""
In-distribution regularization functions.

Author(s):
    Michael Yao

Licensed under the MIT License. Copyright University of Pennsylvania 2023.
"""
import torch
import torch.nn as nn
from typing import Optional, Union


class Regularization(nn.Module):
    """In-distribution regularization functions."""

    def __init__(
        self,
        alpha: float,
        method: Optional[str] = None,
        D: Optional[nn.Module] = None,
        p: Optional[Union[int, str]] = 2
    ):
        """
        Args:
            alpha: regularization penalty weighting term.
            method: method of regularization. One of [`None`, `fid`,
                `importance_weighting`, `_wasserstein_distance`].
            D: source discriminator network. Must be provided if the method of
                regularization is `importance_weighting`.
            p: Wasserstein distance norm. p >= 1 or p == `inf`. Must be
                provided if the method of Regularization is
                `_wasserstein_distance`.
        """
        super().__init__()
        self.alpha = alpha
        self.method = method
        self.D = D
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
        if self.method.lower() == "importance_weighting":
            return self.alpha * (torch.mean(self._importance_weight(xp)) - 1.0)
        elif self.method.lower() == "wasserstein_distance":
            return self.alpha * torch.mean(self._wasserstein_distance(xp, xq))
        elif self.method.lower() == "fid":
            return self.alpha * self._fid(xp, xq)
        else:
            raise NotImplementedError(
                f"Regularization method {self.method} not implemented."
            )

    def _importance_weight(self, xp: torch.Tensor) -> torch.Tensor:
        """
        Computes the estimated importance weights w(x) = q(x) / p(x) given `xp`
        samples from `p(x)` and a source discriminator.
        Input:
            xp: samples from source distribution `p(x)`.
        Returns:
            Estimated importance weights at those values of x.
        """
        # TODO: Should this be (1 / g(x)) - 0.5 as in Sangdon's implementation?
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

    def _fid(self, xp: torch.Tensor, xq: torch.Tensor) -> torch.Tensor:
        """
        Computes the Frechet inception distance between `xp` samples from
        `p(x)` and `xq` samples from `q(x)`.
        Input:
            xp: samples from source distribution `p(x)`.
            xq: samples from target distribution `q(x)`.
        Returns:
            Estimated Frechet inception distance between `p(x)` and `q(x)`.
        """
        mup, muq = torch.mean(xp), torch.mean(xq)
        covarp = torch.cov(xp.view(xp.size(0), -1))
        covarq = torch.cov(xq.view(xq.size(0), -1))
        d2 = torch.square(mup - muq) + torch.trace(
            covarp + covarq - (2.0 * torch.sqrt(covarp * covarq))
        )
        return torch.sqrt(d2)
