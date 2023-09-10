"""
Defines a variational autoencoder (VAE) for molecule generation using SELFIES
representation. Codebase adapted from @aspuru-guzik-group `selfies` repository
at https://github.com/aspuru-guzik-group/selfies/tree/master.

Author(s):
    Michael Yao

Citation(s):
    [1] Krenn M, Haese F, Nigam AK, Friederich P, Aspuru-Guzik A. Self-
        referencing embedded strings (SELFIES): A 100% robust molecular string
        representation. Mach Learn: Sci Tech 1: 045024. (2020).
        https://doi.org/10.1088/2632-2153/aba947

Licensed under the MIT License. Copyright University of Pennsylvania 2023.
"""
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Optional, Sequence

from models.objective import SELFIESObjective
from models.regularization import Regularization

from MolOOD.molformers.models.components import causal_mask
from MolOOD.molformers.models.BaseVAERange import BaseVAE


class SELFIESVAEModule(BaseVAE):
    """VAE model for molecule generation using SELFIES representation."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        vocab: Dict[str, int],
        padding_token: str = "[pad]",
        alpha: float = 0.0,
        KLD_alpha: float = 1e-5,
        regularization: str = "elbo",
        z_dim: int = 64,
        lr: int = 0.0001,
        clip: Optional[float] = None,
        beta1: float = 0.9,
        beta2: float = 0.999,
        dropout_enc: float = 0.1,
        n_critic_per_generator: float = 1.0,
        **kwargs
    ):
        """
        in_dim: input dimensions to the VAE encoder.
        out_dim: output dimensions from the VAE decoder.
        vocab: vocabulary dictionary.
        padding_token: padding token in the vocab dictionary. Default `[pad]`.
        alpha: source critic regularization weighting term. Default no
            source critic regularization.
        KLD_alpha: weighting of KL Divergence in ELBO loss calculation.
        regularization: method of regularization. One of [`None`, `fid`,
            `gan_loss`, `importance_weighting`, `log_importance_weighting`,
            `wasserstein`, `em`, `elbo`].
        z_dim: number of latent space dimensions in the VAE. Default 64.
        lr: learning rate. Default 0.0002.
        clip: gradient clipping. Default no clipping.
        beta1: beta_1 parameter in Adam optimizer algorithm. Default 0.5.
        beta2: beta_2 parameter in Adam optimizer algorithm. Default 0.999.
        dropout_enc: dropout parameters of the encoder. Default 0.1.
        n_critic_per_generator: number of times to optimize the critic
            versus the generator. Default 1.0.
        """
        super().__init__(vocab)
        self.hparams.alpha = min(max(alpha, 0.0), 1.0)
        self.automatic_optimization = False

        self.regularization = None
        if self.hparams.alpha > 0.0:
            self.regularization = Regularization(
                method=regularization, x_dim=(in_dim,), KLD_alpha=KLD_alpha
            )
            self.critic_loss = self.regularization.critic_loss
        else:
            self.hparams.n_critic_per_generator = 1.0

        self.objective = None
        if self.hparams.alpha < 1.0:
            self.objective = SELFIESObjective(
                self.hparams.vocab,
                surrogate_ckpt="./MolOOD/checkpoints/regressor.ckpt"
            )

        if self.hparams.n_critic_per_generator >= 1.0:
            self.f_G, self.f_D = round(self.hparams.n_critic_per_generator), 1
        else:
            self.f_G, self.f_D = 1, round(
                1.0 / self.hparams.n_critic_per_generator
            )

    def forward(self, tokens: torch.Tensor) -> Sequence[torch.Tensor]:
        """
        Forward propagation through the network.
        Input:
            tokens: input batch of token representations of molecules.
        Returns:
            recon_tokens: reconstructed batch of molecules as logits of each
                character in the vocab dictionary.
            mu: encoded mean in the latent space.
            sigma: encoded standard deviation in the latent space.
        """
        mu, sigma = self.encode(tokens)
        z = mu + (sigma * torch.randn_like(sigma))
        return self.decode(z, tokens), mu, sigma

    def decode(self, z: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        """
        TODO
        """
        z = self.dec_neck(z)
        embed = self.decoder_token_embedding(tokens)

        tgt_mask = causal_mask(embed.size(dim=1), embed.device, embed.dtype)
        decoding = self.decoder(tgt=embed, memory=z, tgt_mask=tgt_mask)
        return self.dec_tok_deproj(decoding)

    def training_step(
        self, batch: Sequence[torch.Tensor], batch_idx: int
    ) -> None:
        """
        Training step for generative VAE model.
        Input:
            batch: batch of input data for model training.
            batch_idx: batch index.
        Returns:
            None.
        """
        if self.regularization and self.hparams.regularization != "elbo":
            optimizer_vae, optimizer_D = self.optimizers()
        else:
            optimizer_vae, optimizer_D = self.optimizers(), None

        if batch_idx % self.f_G == 0:
            self.toggle_optimizer(optimizer_vae)
            tokens, _ = batch

            logits, mu, sigma = self(tokens)
            recon_tokens = torch.argmax(logits, dim=-1)

            loss_vae = 0.0
            if self.objective:
                loss_vae += (self.hparams.alpha - 1.0) * torch.mean(
                    self.objective(recon_tokens)
                )
            if self.regularization:
                loss_vae += (self.hparams.alpha) * self.regularization(
                    tokens, logits, mu, sigma
                )
            self.log("loss_vae", loss_vae, prog_bar=True, sync_dist=True)
            self.manual_backward(loss_vae, retain_graph=bool(optimizer_D))
            if self.hparams.clip:
                self.clip_gradients(
                    optimizer_vae,
                    gradient_clip_val=self.hparams.clip,
                    gradient_clip_algorithm="norm"
                )
            optimizer_vae.step()
            optimizer_vae.zero_grad()
            self.untoggle_optimizer(optimizer_vae)

        if optimizer_D and batch_idx % self.f_D == 0:
            self.toggle_optimizer(optimizer_D)
            loss_D = self.critic_loss(batch, recon_tokens.detach())
            self.log("loss_D", loss_D, prog_bar=True, sync_dist=True)
            self.manual_backward(loss_D)
            if self.hparams.clip:
                self.clip_gradients(
                    optimizer_D,
                    gradient_clip_val=self.hparams.clip,
                    gradient_clip_algorithm="norm"
                )
            optimizer_D.step()
            optimizer_D.zero_grad()
            self.untoggle_optimizer(optimizer_D)

        if hasattr(self.regularization, "f"):
            self.regularization.clip()

    def validation_step(
        self, batch: Sequence[torch.Tensor], batch_idx: int
    ) -> None:
        """
        Validation step for generative VAE model.
        Input:
            batch: batch of input data for model training.
            batch_idx: batch index.
        Returns:
            None.
        """
        tokens, _ = batch

        logits, mu, sigma = self(tokens)
        recon_tokens = torch.argmax(logits, dim=-1)

        val_obj = 0.0
        if self.objective:
            val_obj = torch.mean(self.objective(recon_tokens))
        val_reg = 0.0
        if self.regularization:
            val_reg = self.regularization(
                tokens, logits, mu, sigma
            )

        self.log("val_obj", val_obj, prog_bar=True, sync_dist=True)
        self.log("val_reg", val_reg, prog_bar=True, sync_dist=True)

    def configure_optimizers(self) -> Sequence[optim.Optimizer]:
        """
        Configure manual optimization.
        Input:
            None.
        Returns:
            Sequence of optimizer(s).
        """
        vae_params = list(self.encoder.parameters()) + list(
            self.decoder.parameters()
        )

        if self.regularization and (
            self.hparams.regularization in ["wasserstein", "em"]
        ):
            optimizer_vae = optim.RMSprop(vae_params, lr=self.hparams.lr)
            optimizer_D = optim.RMSprop(
                self.regularization.f.parameters(), lr=self.hparams.lr
            )
            return [optimizer_vae, optimizer_D]

        optimizer_vae = optim.Adam(
            vae_params,
            lr=self.hparams.lr,
            betas=(self.hparams.beta1, self.hparams.beta2)
        )
        if self.hparams.regularization == "elbo":
            return [optimizer_vae]

        if self.regularization:
            optimizer_D = optim.Adam(
                self.regularization.D.parameters(),
                lr=self.hparams.lr,
                betas=(self.hparams.beta1, self.hparams.beta2)
            )
            return [optimizer_vae, optimizer_D]
        return [optimizer_vae]


class VAEEncoder(nn.Module):
    """Defines the VAE encoder network as a FCNN."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        layer_features: Optional[Sequence[int]] = [2_048, 1_024, 512]
    ):
        """
        Args:
            in_features: vector length of input.
            out_features: vector length of output.
            layer_features: intermediate feature length(s).
        """
        super().__init__()
        self.features = [in_features] + layer_features + [out_features]

        model = []
        for i in range(len(self.features) - 2):
            in_dim, out_dim = self.features[i], self.features[i + 1]
            model.append((f"linear{i}", nn.Linear(in_dim, out_dim),))
            model.append((f"activation{i}", nn.ReLU(),))
        self.encoder = nn.Sequential(OrderedDict(model))

        self.out_mu = nn.Linear(self.features[-2], self.features[-1])
        self.out_log_var = nn.Linear(self.features[-2], self.features[-1])

    def forward(self, X: torch.Tensor) -> Sequence[torch.Tensor]:
        """
        Forward propagation through the VAE encoder. Uses a scale-location
        transformation to sample from a normal distribution with encoded mean
        `mu` and encoded log of the variance `log_var` in the latent space.
        Input:
            X: input vector to be encoded.
        Returns:
            z: encoded vector in the latent space.
            mu: encoded mean.
            log_var: encoded log of the variance.
        """
        encoder_out = self.encoder(X)
        mu, log_var = self.out_mu(encoder_out), self.out_log_var(encoder_out)
        z = mu + (torch.exp(0.5 * log_var) * torch.randn_like(log_var))
        return z, mu, log_var


class VAEDecoder(nn.Module):
    """Defines the VAE decoder network as an RNN."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_layers: int = 1,
        hidden_features: int = 128
    ):
        """
        Args:
            in_features: vector length of input.
            out_features: vector length of output.
            num_layers: number of layers of the RNN.
            hidden_features: number of features in the hidden state in the RNN.
        """
        super().__init__()

        self.hidden_features = hidden_features
        self.num_layers = num_layers
        self.decoder = nn.GRU(
            input_size=in_features,
            hidden_size=hidden_features,
            num_layers=num_layers,
            batch_first=False
        )
        self.linear = nn.Linear(hidden_features, out_features)

    def forward(
        self, z: torch.Tensor, h0: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward propagation through the VAE decoder.
        Input:
            z: input latent space vector to be decoded.
            h0: optional initial hidden state. Defaults to zeros.
        Returns:
            Decoded vector.
        """
        output, hn = self.decoder(z, h0)
        return self.linear(output)

    def init_hidden(self, batch_size: int) -> torch.Tensor:
        """
        Initialize hidden layer.
        Input:
            batch_size: batch size.
        Returns:
            Tensor for hidden layer initialization.
        """
        return next(self.parameters()).new_zeros(
            self.num_layers, batch_size, self.hidden_features
        )
