"""
Conditional tabular GAN for counterfactual patient data generation.

Author(s):
    Michael Yao @michael-s-yao

Citation(s):
    [1] Xu Lei, Skoularidou M, Cuesta-Infante A, Veeramachaneni K. Modeling
        tabular data using conditional GAN. Proc NeurIPS. (2019).
        https://doi.org/10.48550/arXiv.1907.00503
    [2] CTGAN Github repo from @sdv-dev at https://github.com/sdv-dev/CTGAN

Licensed under the MIT License. Copyright University of Pennsylvania 2023.
"""
from collections import defaultdict
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import lightning.pytorch as pl
from typing import Sequence, Union

from data.iwpc import PatientSample
from models.fcnn import FCNN
from models.mortality_estimator import WarfarinMortalityLightningModule


class SourceCritic(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dims: Sequence[int],
        dropout: float,
        pac: int = 16
    ):
        """
        Args:
            in_dim: dimensions of input data.
            hidden_dims: dimensions of the hidden intermediate layers.
            dropout: dropout. Default 0.1.
            pac: number of samples to pack together according to the PacGAN
                framework. Default 16.
        Citation(s):
            [1] Lin Z, Khetan A, Fanti G, Oh S. PacGAN: The power of two
                samples in generative adversarial networks. arXiv. (2017).
                https://doi.org/10.48550/arXiv.1712.04086
        """
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.pac = pac

        self.model = FCNN(
            in_dim=(self.in_dim * self.pac),
            out_dim=1,
            hidden_dims=self.hidden_dims,
            dropout=self.dropout,
            final_activation=None
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the source critic network.
        Input:
            X: input tensor of shape BN, where B is the batch size and
                N is the number of input dimensions.
        Returns:
            An output vector of shape B equal to SourceCritic(X).
        """
        return torch.mean(self.model(X.view(-1, self.in_dim * self.pac)))

    def gradient_penalty(
        self, real: torch.Tensor, generated: torch.Tensor, K: int = 1
    ) -> None:
        """
        Calculates the gradient penalty term to keep the critic K-Lipschitz.
        Input:
            real: samples from the true distribution.
            generated: samples from the generated distribution.
            K: Lipschitz constant. Default 1.
        Returns:
            Quadratic penalty term enforcing the critic to be K-Lipschitz.
        """
        mix = torch.rand(real.size(dim=-1)).to(real)
        X = (mix * real) + ((1 - mix) * generated)
        X.requires_grad_(True)
        y = self(X)
        gradX = torch.autograd.grad(
            outputs=y,
            inputs=X,
            grad_outputs=torch.ones_like(y),
            retain_graph=True,
            create_graph=True,
            only_inputs=True
        )
        gradX = gradX[0].view(-1, self.pac * real.size(dim=1))
        return torch.mean(torch.square(torch.norm(gradX, p=2, dim=1) - K))


class Residual(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        """
        Args:
            in_dim: dimensions of input data.
            out_dim: dimensions of output data.
        """
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim - in_dim
        self.model = nn.Sequential(
            nn.Linear(self.in_dim, self.out_dim),
            nn.BatchNorm1d(self.out_dim),
            nn.ReLU()
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the residual layer.
        Input:
            X: input tensor of shape BN, where B is the batch size and
                N is the number of input dimensions.
        Returns:
            An output tensor of shape BM, where B is the batch size and
                M is the total number of input and output dimensions.
        """
        return torch.cat((self.model(X), X), dim=-1)


class Generator(nn.Module):
    def __init__(
        self, in_dim: int, hidden_dims: Sequence[int], data_dim: int
    ):
        """
        Args:
            in_dim: size of the input vector to the generator.
            hidden_dims: size of the output samples for each of the Residual
                layers.
        """
        super().__init__()
        self.dims = np.cumsum([in_dim] + hidden_dims)
        self.data_dim = data_dim
        self.model = [
            Residual(self.dims[i], self.dims[i + 1])
            for i in range(len(self.dims) - 1)
        ]
        self.model.append(nn.Linear(self.dims[-1], self.data_dim))
        self.model = nn.Sequential(*self.model)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the generator network.
        Input:
            z: input tensor of shape BN, where B is the batch size and N is the
                number of embedding dimensions.
        Returns:
            An output tensor of shape BM, where B is the batch size and M is
                the output dimension `data_dim`.
        """
        return self.model(z)


class CTGANLightningModule(pl.LightningModule):
    def __init__(
        self,
        patient_vector_dim: int,
        condition_mask_dim: int,
        alpha: float,
        lambda_: float = 10,
        embedding_dim: int = 64,
        generator_dims: Sequence[int] = [256, 256],
        critic_dims: Sequence[int] = [512, 512],
        optimizer: str = "Adam",
        lr: Union[float, Sequence[float]] = 0.0002,
        weight_decay: Union[float, Sequence[float]] = 1e-6,
        dropout: float = 0.1,
        batch_size: int = 64,
        n_critic_per_generator: int = 5,
        cost_ckpt: Union[Path, str] = "./ckpts/warfarin_cost_estimator.ckpt",
        wasserstein: bool = True
    ):
        """
        Args:
            patient_vector_dim: length of the patient input vector.
            condition_mask_dim: length of the condition mask.
            alpha: source critic regularization weighting term.
            lambda_: source critic gradient penalty weighting term. Default 10.
            embedding_dim: size of the random sample inputs to the generator.
            generator_dims: size of the output samples for each of the Residual
                layers in the generator model.
            critic_dims: dimensions of the hidden intermediate layers in the
                source critic model.
            optimizer: optimizer. Default `Adam`.
            lr: learning rate. Default 0.0002 for both the generator and the
                source critic.
            weight_decay: weight decay. Default 1e-6 for both the generator and
                the source critic.
            dropout: dropout parameter for the source critic. Default 0.1.
            batch_size: batch size. Default 64.
            n_critic_per_generator: number of times to optimize the critic
                versus the generator. Default 5.
            cost_ckpt: checkpoint to the trained Warfarin cost estimator
                function.
            wasserstein: whether to train a Wasserstein GAN instead of a
                traditional GAN.
        """
        super().__init__()
        self.save_hyperparameters()
        self.hparams.alpha = min(max(self.hparams.alpha, 0.0), 1.0)
        self.automatic_optimization = False

        in_dim_generator = self.hparams.embedding_dim
        in_dim_generator += self.hparams.condition_mask_dim
        self.generator = Generator(
            in_dim=in_dim_generator,
            hidden_dims=self.hparams.generator_dims,
            data_dim=self.hparams.patient_vector_dim
        )

        self.critic = SourceCritic(
            in_dim=self.hparams.patient_vector_dim,
            hidden_dims=self.hparams.critic_dims,
            dropout=self.hparams.dropout
        )

        self.cost = WarfarinMortalityLightningModule.load_from_checkpoint(
            self.hparams.cost_ckpt
        )
        self.cost.eval()

    def forward(self, batch: PatientSample) -> torch.Tensor:
        """
        Forward pass through the generative model.
        Input:
            batch: an input batch as a PatientSample object.
        Returns:
            A batch of `batch_size` generated samples.
        """
        mean = torch.zeros((batch.X.size(dim=0), self.hparams.embedding_dim))
        mean = mean.to(self.device)
        std = torch.ones_like(mean)
        z = torch.normal(mean, std)
        z = torch.cat((z, batch.cond_mask), dim=-1)
        return self.generator(z)

    def _gumbel_softmax(
        self, logits: torch.Tensor, tau: float, num_attempts: int = 10
    ) -> torch.Tensor:
        """
        Applies the Gumbel-Softmax function to a series of logits.
        Input:
            logits: unnormalized log probabilities.
            tau: scalar temperature parameter.
        Returns:
            Sampled tensor from the Gumbel-Softmax distribution.
        """
        for _ in range(10):
            y = F.gumbel_softmax(logits, tau=tau, dim=-1)
            if not torch.any(torch.isnan(y)):
                return y
        raise ValueError(
            f"Gumbel-Softmax function returning NaN for input logits {logits}"
        )

    def training_step(self, batch: PatientSample, batch_idx: int) -> None:
        """
        Training step for the generative model.
        Input:
            batch: batch of input data for model training.
            batch_idx: batch index.
        Returns:
            None.
        """
        generated = self._activate(self(batch), batch.X_attributes)
        generator_optimizer, critic_optimizer = self.optimizers()

        if batch_idx % self.hparams.n_critic_per_generator == 0:
            self.toggle_optimizer(generator_optimizer)
            if self.hparams.wasserstein:
                generator_loss = -self.hparams.alpha * self.critic(generated)
            else:
                generator_loss = -self.hparams.alpha * torch.log(
                    torch.sigmoid(self.critic(generated))
                )
            cost = (1.0 - self.hparams.alpha) * torch.mean(
                self.cost(generated)
            )
            generator_penalty = self._generator_penalty(
                batch.X, generated, batch.X_attributes
            )
            self.manual_backward(generator_penalty, retain_graph=True)
            self.manual_backward(cost, retain_graph=True)
            self.manual_backward(generator_loss, retain_graph=True)
            generator_optimizer.step()
            generator_optimizer.zero_grad()
            self.untoggle_optimizer(generator_optimizer)

        self.toggle_optimizer(critic_optimizer)
        if self.hparams.wasserstein:
            critic_loss = self.critic(generated.detach()) - self.critic(
                batch.X
            )
        else:
            critic_loss = torch.log(
                torch.sigmoid(self.critic(generated.detach()))
            )
            critic_loss -= torch.log(torch.sigmoid(self.critic(batch.X)))
        penalty_loss = self.hparams.lambda_ * self.critic.gradient_penalty(
            batch.X, generated.detach()
        )
        self.manual_backward(penalty_loss, retain_graph=True)
        self.manual_backward(critic_loss)
        critic_optimizer.step()
        critic_optimizer.zero_grad()
        self.untoggle_optimizer(critic_optimizer)

    def _activate(
        self, X: torch.Tensor, X_attributes: Sequence[str]
    ) -> torch.Tensor:
        """
        Applies the correct activation function to generated tensor components.
        Input:
            X: an input tensor generated by the generator.
            X_attributes: a sequence of the attribute names of each dimension.
        Returns:
            X after the correct activation function has been applied to each
            element.
        """
        num_classes = defaultdict(lambda: 0)
        for attr in X_attributes:
            if attr == "Therapeutic Dose of Warfarin":
                continue
            key = attr.split("_")[0].replace(".component", "")
            num_classes[key] += 1

        idx = 0
        for attr, number in num_classes.items():
            if attr.endswith(".normalized"):
                X[:, idx] = torch.tanh(X[:, idx])
                idx += 1
                continue
            end = idx + number
            X[:, idx:end] = self._gumbel_softmax(X[:, idx:end], tau=0.2)
            idx = end
        return X

    def _generator_penalty(
        self, Xp: torch.Tensor, Xq: torch.Tensor, X_attributes: torch.Tensor
    ) -> torch.Tensor:
        """
        Log likelihood penalty loss term for the generator.
        Input:
            Xp: a batch of rows drawn from the true distribution.
            Xq: a batch of generated rows.
            X_attributes: a sequence of the attribute names of each dimension.
        Returns:
            The value of the generator penalty term.
        """
        penalty_loss = 0.0
        num_classes = defaultdict(lambda: 0)
        for attr in X_attributes:
            if attr == "Therapeutic Dose of Warfarin":
                continue
            elif attr.endswith(".normalized"):
                continue
            key = attr.split("_")[0].replace(".component", "")
            num_classes[key] += 1

        idx = 0
        for attr, number in num_classes.items():
            end = idx + number
            penalty_loss += F.nll_loss(
                Xq[:, idx:end],
                target=torch.argmax(Xq[:, idx:end], dim=-1),
                reduce="sum"
            )
            idx = end
        return penalty_loss / Xq.size(0)

    def configure_optimizers(self) -> Sequence[optim.Optimizer]:
        """
        Configure manual optimization.
        Input:
            None.
        Returns:
            Sequence of optimizer(s).
        """
        if isinstance(self.hparams.lr, float):
            lr_generator, lr_critic = self.hparams.lr, self.hparams.lr
        else:
            lr_generator, lr_critic = self.hparams.lr

        if isinstance(self.hparams.weight_decay, float):
            decay_generator = self.hparams.weight_decay
            decay_critic = self.hparams.weight_decay
        else:
            decay_generator, decay_critic = self.hparams.weight_decay

        if self.hparams.optimizer.title() == "Adam":
            return [
                optim.Adam(
                    self.generator.parameters(),
                    lr=lr_generator,
                    weight_decay=decay_generator
                ),
                optim.Adam(
                    self.critic.parameters(),
                    lr=lr_critic,
                    weight_decay=decay_critic
                )
            ]
        elif self.hparams.optimizer.upper() == "SGD":
            return [
                optim.SGD(
                    self.generator.parameters(),
                    lr=lr_generator,
                    weight_decay=decay_generator
                ),
                optim.SGD(
                    self.critic.parameters(),
                    lr=lr_critic,
                    weight_decay=decay_critic
                )
            ]
        elif self.hparams.optimizer.lower() == "rmsprop":
            return [
                optim.RMSprop(
                    self.generator.parameters(),
                    lr=lr_generator,
                    weight_decay=decay_generator
                ),
                optim.RMSprop(
                    self.critic.parameters(),
                    lr=lr_critic,
                    weight_decay=decay_critic
                )
            ]
        else:
            raise NotImplementedError(
                f"Unrecognized optimizer {self.hparams.optimizer} requested."
            )
