"""
In-distribution regularization functions.

Author(s):
    Michael Yao

Licensed under the MIT License. Copyright University of Pennsylvania 2023.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Sequence, Tuple

from models.critic import Critic, WeightClipper
from models.rnn import RNN


class Regularization(nn.Module):
    """In-distribution regularization functions."""

    def __init__(
        self,
        method: Optional[str] = None,
        x_dim: Optional[Tuple[int]] = (1, 28, 28),
        intermediate_layers: Sequence[int] = (512, 256),
        c: Optional[float] = 0.01,
        KLD_alpha: Optional[float] = 1e-5,
        use_rnn: bool = False,
        **kwargs
    ):
        """
        Args:
            method: method of regularization. One of [`None`, `gan_loss`,
                `importance_weighting`, `log_importance_weighting`,
                `wasserstein`, `em`, `elbo`].
            x_dim: dimensions CHW of the output image from the generator G.
                Default MNIST dimensions (1, 28, 28).
            intermediate_layers: intermediate layer output dimensions. Default
                (512, 256).
            c: weight clipping to enforce 1-Lipschitz condition on source
                critic for `wasserstein`/`em` regularization algorithms.
            KLD_alpha: weighting of KL Divergence loss term, only applicable
                if `method` is `elbo`. Default 1e-5.
            use_rnn: whether to use an RNN architecture for the source critic.
        """
        super().__init__()
        self.method = method.lower() if method else method
        self.use_rnn = use_rnn
        if self.method in [
            "gan_loss", "importance_weighting", "log_importance_weighting"
        ]:
            if self.use_rnn:
                self.D = RNN(
                    cell_type="GRU",
                    out_dim=1,
                    vocab=kwargs["vocab"],
                    num_dimensions=kwargs.get("num_dimensions", 128),
                    num_layers=kwargs.get("num_layers", 3),
                    embedding_layer_size=kwargs.get(
                        "embedding_layer_size", 128
                    ),
                    device=kwargs.get("device", "cpu"),
                    dropout=kwargs.get("dropout", 0.1),
                    use_bidirectional=True,
                    use_sigmoid=True,
                    return_hidden=False
                )
            else:
                self.D = Critic(
                    x_dim=x_dim,
                    intermediate_layers=intermediate_layers,
                    use_sigmoid=True
                )
        elif self.method in ["wasserstein", "em"]:
            if self.use_rnn:
                self.f = RNN(
                    cell_type="GRU",
                    out_dim=1,
                    vocab=kwargs["vocab"],
                    num_dimensions=kwargs.get("num_dimensions", 128),
                    num_layers=kwargs.get("num_layers", 3),
                    embedding_layer_size=kwargs.get(
                        "embedding_layer_size", 128
                    ),
                    device=kwargs.get("device", "cpu"),
                    dropout=kwargs.get("dropout", 0.1),
                    use_bidirectional=True,
                    use_sigmoid=False,
                    return_hidden=False
                )
            else:
                self.f = Critic(
                    x_dim=x_dim,
                    intermediate_layers=intermediate_layers,
                    use_sigmoid=False
                )
            self.clipper = WeightClipper(c=c)
            self.clip()
        elif self.method == "elbo":
            self.KLD_alpha = KLD_alpha

    def forward(
        self,
        xp: torch.Tensor,
        xq: Optional[torch.Tensor] = None,
        mu: Optional[torch.Tensor] = None,
        log_var: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward propagation through a regularization function.
        Input:
            xp: samples from the source distribution `p(x)`.
            xq: samples from the target distribution `q(x)`.
            mu: encoded mean in the latent space. Only applicable to `elbo`
                regularization method.
            log_var: encoded log of the variance in the latent space. Only
                applicable to `elbo` regularization method.
        Returns:
            In-distribution regularization loss term.
        """
        if self.method is None or self.method == "":
            return 0.0

        if self.method == "gan_loss":
            return torch.mean(self._gan_loss(xq))
        elif self.method == "importance_weighting":
            w_xp = torch.mean(torch.square(self._importance_weight(xp) - 1.0))
            w_xq = torch.mean(torch.square(self._importance_weight(xq) - 1.0))
            return 0.5 * (w_xp + w_xq)
        elif self.method == "log_importance_weighting":
            log_w_xp = 2.0 * torch.mean(
                torch.log(torch.abs(1.0 - (2.0 * self.D(xp)))) - torch.log(
                    self.D(xp)
                )
            )
            log_w_xq = 2.0 * torch.mean(
                torch.log(torch.abs(1.0 - (2.0 * self.D(xq)))) - torch.log(
                    self.D(xq)
                )
            )
            return 0.5 * (log_w_xp + log_w_xq)
        elif self.method in ["wasserstein", "em"]:
            return self._wasserstein_distance_1(xp, xq)
        elif self.method == "elbo":
            return self._elbo(xp, xq, mu, log_var)
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

    def _importance_weight(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the estimated importance weights w(x) = q(x) / p(x).
        Input:
            x: inputs at which to estimate the importance weight.
        Returns:
            Estimated importance weights - 1.0 at those values of x.
        """
        return (1.0 / self.D(x)) - 1.0

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

    def _elbo(
        self,
        xp: torch.Tensor,
        xq: torch.Tensor,
        mu: torch.Tensor,
        log_var: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculates the ELBO loss.
        Input:
            xp: original input tensor to the VAE from the source distribution
                `p(x)`.
            xq: reconstructed tensor output from the VAE from the target
                distribution `q(x)`.
            mu: encoded mean in the latent space.
            log_var: encoded log of the variance in the latent space.
        Returns:
            Calculated ELBO loss.
        """
        xp = torch.argmax(xp.view(-1, xp.size(-1)), dim=-1)
        xq = xq.view(-1, xq.size(-1))
        recon_loss = F.cross_entropy(xq, xp)
        kld = -0.5 * torch.mean(1.0 + log_var - (mu * mu) - torch.exp(log_var))
        return recon_loss + (self.KLD_alpha * kld)

    def critic_loss(
        self, xp: torch.Tensor, xq: torch.Tensor
    ) -> Optional[torch.Tensor]:
        """
        Source critic loss function implementation.
        Input:
            xp: samples from source distribution `p(x)`.
            xq: samples from target distribution `q(x)`.
        Returns:
            Source critic loss L(xp, xq).
        """
        if self.method in [
            "gan_loss", "importance_weighting", "log_importance_weighting"
        ]:
            return torch.mean(
                -1.0 * (torch.log(self.D(xp)) + torch.log(1.0 - self.D(xq)))
            )
        elif self.method in ["wasserstein", "em"]:
            return -1.0 * self._wasserstein_distance_1(xp, xq)
        else:
            return None

    def clip(self) -> torch.Tensor:
        """
        Clips the weights of self.f to [-self.clipper, self.clipper].
        Input:
            None.
        Returns:
            None.
        """
        self.f.apply(self.clipper)
