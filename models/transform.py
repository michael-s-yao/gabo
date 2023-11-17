import numpy as np
import torch


class SobelGaussianTransform:
    """
    Uses the Box-Muller transform to transform a Sobel engine-drawn
    distribution into a Gaussian distribution with zero mean and unit variance.
    """
    def __init__(self, mu: torch.Tensor, sigma: torch.Tensor):
        self.mu, self.sigma = mu.item(), sigma.item()

    def __call__(self, sobel: torch.Tensor) -> torch.Tensor:
        """
        Uses the Box-Muller transform to transform a Sobel engine-drawn
        distribution into a Gaussian distribution with zero mean and unit
        variance.
        Input:
            sobel: observations drawn according to a Sobel engine.
        Returns:
            The transformed points conforming to a Gaussian distribution.
        """
        m = sobel.size(dim=-1) // 2
        u1, u2 = sobel[..., :m], sobel[..., m:]
        z1 = torch.sqrt(-2.0 * torch.log(u1)) * torch.cos(2.0 * np.pi * u2)
        z2 = torch.sqrt(-2.0 * torch.log(u1)) * torch.sin(2.0 * np.pi * u2)
        return (self.mu + (self.sigma * torch.hstack([z1, z2]))).to(sobel)


class GaussianScalingTransform:
    """
    Scales a Gaussian distribution to zero mean and unit variance.
    """
    def __init__(self, mu: torch.Tensor, sigma: torch.Tensor):
        self.mu, self.sigma = mu.item(), sigma.item()

    def __call__(self, gaussian: torch.Tensor) -> torch.Tensor:
        """
        Scales a Gaussian distribution to zero mean and unit variance.
        Input:
            gaussian: observations drawn from a Gaussian distribution with mean
                mu and standard deviation sigma.
        Returns:
            The transformed points conforming to a Guassian distribution with
            zero mean and unit variance.
        """
        return (gaussian - self.mu) / self.sigma

    def invert(self, gaussian: torch.Tensor) -> torch.Tensor:
        """
        Scales a Gaussian distribution with zero mean and unit variance to
        mean mu and standard deviation sigma.
        Input:
            gaussian: observations drawn from a Gaussian distribution with
                zero mean and unit variance.
        Returns:
            The transformed points conforming to a Gaussian distribution with
                mean mu and standard deviation sigma.
        """
        return self.mu + (self.sigma * gaussian)
