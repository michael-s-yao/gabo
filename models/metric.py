"""
Metrics for model training and/or evaluation.

Author(s):
    Michael Yao

Licensed under the MIT License. Copyright University of Pennsylvania 2023.
"""
import torch


def FID(Xp: torch.Tensor, Xq: torch.Tensor) -> torch.Tensor:
    """
    Computes the Frechet inception distance between `Xp` samples from
    `p(x)` and `Xq` samples from `q(x)`.
    Input:
        Xp: samples from source distribution `p(x)`.
        Xq: samples from target distribution `q(x)`.
    Returns:
        Estimated Frechet inception distance between `p(x)` and `q(x)`.
    """
    mup, muq = torch.mean(Xp), torch.mean(Xq)
    covarp = torch.cov(Xp.reshape(Xp.size(0), -1))
    covarq = torch.cov(Xq.reshape(Xq.size(0), -1))
    d2 = torch.square(mup - muq) + torch.trace(
        covarp + covarq - (2.0 * torch.sqrt(covarp * covarq))
    )
    return torch.sqrt(d2)
