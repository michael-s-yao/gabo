"""
Metrics for model training and/or evaluation.

Author(s):
    Michael Yao

Licensed under the MIT License. Copyright University of Pennsylvania 2023.
"""
import torch


def FID(xp: torch.Tensor, xq: torch.Tensor) -> torch.Tensor:
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
