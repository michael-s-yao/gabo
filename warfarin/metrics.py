"""
Utility functions for evaluation of warfarin counterfactual generation.

Author(s):
    Michael Yao @michael-s-yao

Licensed under the MIT License. Copyright University of Pennsylvania 2023.
"""
import numpy as np


def Divergence(
    P: np.ndarray,
    Q: np.ndarray,
    n: int = 20,
    symmetric: bool = True,
    eps: float = 1e-12
) -> float:
    """
    Calculates the divergence between two vectors of values P and Q.
    Input:
        P: a vector of values from the target distribution.
        Q: a vector of values from the source generated distribution.
        n: number of bins. Default 20.
        symmetric: if True, the JS divergence is calculated. Otherwise, the
            KL divergence is calculated instead.
        eps: a small value to add to values to avoid division by zero.
    Returns:
        D(P || Q).
    """
    bins = np.linspace(np.min(P), np.max(P), num=n)
    bins = [-np.inf] + bins.tolist() + [np.inf]
    p_, q_ = np.zeros(len(bins) - 1), np.zeros(len(bins) - 1)
    for i in range(len(bins) - 1):
        p_[i] = len(set(np.where(bins[i] <= P)[0].tolist()) & set(
            np.where(P < bins[i + 1])[0].tolist()
        ))
        q_[i] = len(set(np.where(bins[i] <= Q)[0].tolist()) & set(
            np.where(Q < bins[i + 1])[0].tolist()
        ))
    p_, q_ = p_ / np.sum(p_), q_ / np.sum(q_)
    if symmetric:
        m_ = 0.5 * (p_ + q_)
        a = np.where(
            (p_ > 0) & (m_ > 0), p_ * np.log2((p_ + eps) / (m_ + eps)), 0
        )
        b = np.where(
            (q_ > 0) & (m_ > 0), q_ * np.log2((q_ + eps) / (m_ + eps)), 0
        )
        return 0.5 * (np.sum(a) + np.sum(b))
    return np.sum(
        np.where((p_ > 0) & (q_ > 0), p_ * np.log2((p_ + eps) / (q_ + eps)), 0)
    )


def SupportCoverage(P: np.ndarray, Q: np.ndarray) -> float:
    """
    Calculates the support coverage of Q compared to P.
    Input:
        P: a vector of values from the target distribution.
        Q: a vector of values from the source generated distribution.
    Returns:
        The support coverage of Q compared to P.
    """
    num = min(np.max(P), np.max(Q)) - max(np.min(P), np.min(Q))
    return num / (np.max(P) - np.min(P))
