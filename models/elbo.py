"""
Evidence lower bound (ELBO) implementation for VAE training.

Author(s):
    Michael Yao

Licensed under the MIT License. Copyright University of Pennsylvania 2023.
"""
import torch
import torch.nn as nn


class ELBO(nn.Module):
    """ELBO implementation for VAE training."""

    def __init__(self, KLD_alpha: float = 1e-5):
        """
        Args:
            KLD_alpha: weighting of KL Divergence loss term.
        """
        self.KLD_alpha = KLD_alpha
        self.recon_loss = nn.CrossEntropyLoss()

    def forward(
        self,
        X: torch.Tensor,
        Xhat: torch.Tensor,
        mu: torch.Tensor,
        log_var: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculates the ELBO loss.
        Input:
            X: original input tensor to the VAE.
            Xhat: reconstructed tensor output from the VAE.
            mu: encoded mean in the latent space.
            log_var: encoded log of the variance in the latent space.
        Returns:
            Calculated ELBO loss.
        """
        recon_loss = self.recon_loss(
            Xhat.reshape(-1, Xhat.size(2)),
            torch.argmax(X.reshape(-1, X.size(2)), dim=1)
        )
        kld = -0.5 * torch.mean(1.0 + log_var - (mu * mu) - torch.exp(log_var))
        return recon_loss + (self.KLD_alpha * kld)
