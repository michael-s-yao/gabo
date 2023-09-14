"""
SeqGAN generative network implementation.

Author(s):
    Michael Yao

Citation(s):
    [1] Yu L, Zhang W, Wang J, Yu Y. SeqGAN: Sequence generative adversarial
        nets with policy gradient. AAAI. (2017). https://doi.org/10.48550/arXi
        v.1609.05473

Licensed under the MIT License. Copyright University of Pennsylvania 2023.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import lightning.pytorch as pl
from typing import Dict, Optional, Sequence

from data.molecule import one_hot_encodings_to_tokens
from models.objective import SELFIESObjective
from models.regularization import Regularization


class SeqGANGenerator(nn.Module):
    """SeqGAN generative network implementation as an RNN."""

    def __init__(
        self,
        vocab: Dict[str, int],
        embedding_dim: int,
        hidden_dim: int,
        max_molecule_length: int = 109,
        padding_token: Optional[str] = "[pad]"
    ):
        """
        Args:
            vocab: dictionary of vocabulary.
            embedding_dim: the size of each embedding vector.
            hidden_dim: dimensions of the hidden vector in the RNN.
            max_molecule_length: maximum molecule length. Default 109.
            padding_token: optional padding token. Default `[pad]`.
        """
        super().__init__()
        self.vocab = vocab
        self.num_embeddings = len(self.vocab.keys())
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.max_molecule_length = max_molecule_length
        self.padding_token = padding_token

        self.embed = nn.Embedding(
            self.num_embeddings,
            self.embedding_dim,
            padding_idx=self.vocab[self.padding_token]
        )
        self.rnn = nn.LSTM(
            self.embedding_dim, self.hidden_dim, batch_first=True
        )
        self.w = nn.Linear(self.hidden_dim, self.num_embeddings)
        self.init_params()

    def init_hidden(self, batch_size: int) -> Sequence[torch.Tensor]:
        """
        Initializes the hidden and cell states of the LSTM.
        Input:
            batch_size: batch size.
        Returns:
            hidden: initial hidden state.
            cell: initial cell state.
        """
        hidden = torch.zeros((1, batch_size, self.hidden_dim))
        cell = torch.zeros((1, batch_size, self.hidden_dim))
        return hidden, cell

    def init_params(self, eps: float = 0.05) -> None:
        """
        Initializes the model parameters to have values in [-eps, eps].
        Input:
            eps: upper bound on the magnitude of any model parameter during
                initialization.
        Returns:
            None.
        """
        for param in self.parameters():
            param.data.uniform_(-eps, eps)

    def forward(
        self,
        X: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
        cell: Optional[torch.Tensor] = None,
    ) -> Sequence[torch.Tensor]:
        """
        Forward pass through the generator network.
        Input:
            X: input into the network.
            hidden: input hidden state for the LSTM. Defaults to all zeros.
            cell: input cell state for the LSTM. Defaults to all zeros.
        Returns:
            Output from the network in addition to the new hidden and cell
                states.
        """
        batch_size, seq_length = X.size()
        if hidden is None or cell is None:
            hidden, cell = self.init_hidden(batch_size)
        Y, (hidden, cell) = self.rnn(self.embed(X.long()), (hidden, cell))
        return torch.squeeze(F.softmax(self.w(Y), dim=-1), dim=1), hidden, cell

    def sample(self, num_samples: int) -> torch.Tensor:
        """
        Sample a batch of generated molecules.
        Input:
            num_samples: number of molecules to generate.
        Returns:
            A batch of molecules represented as a tensor with dimensions
            num_samples x max_molecule_length.
        """
        samples = torch.zeros(num_samples, self.max_molecule_length)
        tokens, hidden, cell = torch.zeros(num_samples, 1).long(), None, None
        for i in range(self.max_molecule_length):
            probs, hidden, cell = self(tokens, hidden, cell)
            tokens = torch.multinomial(probs, 1)
            samples[:, i] = torch.squeeze(tokens, dim=-1)
        return samples


class SeqGANGeneratorModule(pl.LightningModule):
    """
    Molecule generation to optimize against an objective with optional source
    critic regularization.
    """

    def __init__(
        self,
        vocab: Dict[str, int],
        embedding_dim: int = 64,
        hidden_dim: int = 64,
        max_molecule_length: int = 109,
        alpha: float = 0.0,
        regularization: str = "gan_loss",
        padding_token: Optional[str] = "[pad]",
        lr: float = 0.0001,
        clip: Optional[float] = None,
        beta1: float = 0.5,
        beta2: float = 0.999,
        n_critic_per_generator: float = 1.0,
        **kwargs
    ):
        """
        Args:
            vocab: vocab dictionary.
            embedding_dim: the size of each embedding vector.
            hidden_dim: dimensions of the hidden vector in the RNN.
            max_molecule_length: maximum molecule length. Default 109.
            alpha: source critic regularization weighting term. Default no
                source critic regularization.
            regularization: method of regularization. One of [`None`, `fid`,
                `gan_loss`, `importance_weighting`, `log_importance_weighting`,
                `wasserstein`, `em`].
            padding_token: optional padding token. Default `[pad]`.
            lr: learning rate. Default 0.0001.
            clip: gradient clipping. Default no clipping.
            beta1: beta_1 parameter in Adam optimizer algorithm. Default 0.5.
            beta2: beta_2 parameter in Adam optimizer algorithm. Default 0.999.
            n_critic_per_generator: number of times to optimize the critic
                versus the generator. Default 1.0.
        """
        super().__init__()
        self.save_hyperparameters()
        self.hparams.alpha = min(max(alpha, 0.0), 1.0)
        self.automatic_optimization = False

        self.generator = SeqGANGenerator(
            vocab,
            embedding_dim=self.hparams.embedding_dim,
            hidden_dim=self.hparams.hidden_dim,
            max_molecule_length=self.hparams.max_molecule_length,
            padding_token=self.hparams.padding_token
        )

        self.regularization = None
        if self.hparams.alpha > 0.0:
            self.regularization = Regularization(
                method=regularization,
                x_dim=self.hparams.x_dim
            )
            self.critic_loss = self.regularization.critic_loss
        else:
            self.hparams.n_critic_per_generator = 1.0

        self.objective = None
        if self.hparams.alpha < 1.0:
            self.objective = SELFIESObjective(
                self.hparams.vocab,
                surrogate_ckpt="./MolOOD/checkpoints/regressor.ckpt",
            )

        if self.hparams.n_critic_per_generator >= 1.0:
            self.f_G, self.f_D = round(self.hparams.n_critic_per_generator), 1
        else:
            self.f_G, self.f_D = 1, round(
                1.0 / self.hparams.n_critic_per_generator
            )

    def forward(self, batch_size: int) -> torch.Tensor:
        """
        Forward propagation through the network.
        Input:
            num_samples: number of molecules to generate.
        Returns:
            A batch of molecules represented as a tensor with dimensions
            num_samples x max_molecule_length.
        """
        return self.generator.sample(batch_size)

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        """
        Training step for generative model.
        Input:
            batch: batch of input data for model training.
            batch_idx: batch index.
        Returns:
            None.
        """
        xp = one_hot_encodings_to_tokens(batch)
        batch_size, seq_length, vocab_size = batch.size()

        if self.regularization:
            optimizer_G, optimizer_D = self.optimizers()
        else:
            optimizer_G, optimizer_D = self.optimizers(), None

        xq = self(batch_size).long()

        if batch_idx % self.f_G == 0:
            self.toggle_optimizer(optimizer_G)

            loss_G = 0.0
            if self.objective:
                loss_G += (self.hparams.alpha - 1.0) * torch.mean(
                    self.objective(xq)
                )
            if self.regularization:
                loss_G += (self.hparams.alpha) * self.regularization(xp, xq)
            self.log("loss_G", loss_G, prog_bar=True, sync_dist=True)
            self.manual_backward(loss_G, retain_graph=bool(optimizer_D))
            if self.hparams.clip:
                self.clip_gradients(
                    optimizer_G,
                    gradient_clip_val=self.hparams.clip,
                    gradient_clip_algorithm="norm"
                )
            optimizer_G.step()
            optimizer_G.zero_grad()
            self.untoggle_optimizer(optimizer_G)

        if optimizer_D and batch_idx % self.f_D == 0:
            self.toggle_optimizer(optimizer_D)
            loss_D = self.critic_loss(xp, xq.detach())
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

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        """
        Validation step for generative model.
        Input:
            batch: batch of input data for model training.
            batch_idx: batch index.
        Returns:
            None.
        """
        pass  # TODO

    def test_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        """
        Testing step for generative model.
        Input:
            batch: batch of input data for model training.
            batch_idx: batch index.
        Returns:
            None.
        """
        pass  # TODO

    def configure_optimizers(self) -> Sequence[optim.Optimizer]:
        """
        Configure manual optimization.
        Input:
            None.
        Returns:
            Sequence of optimizer(s).
        """
        if self.regularization and (
            self.hparams.regularization in ["wasserstein", "em"]
        ):
            optimizer_G = optim.RMSprop(
                self.generator.parameters(), lr=self.hparams.lr
            )
            optimizer_D = optim.RMSprop(
                self.regularization.f.parameters(), lr=self.hparams.lr
            )
            return [optimizer_G, optimizer_D]

        optimizer_G = optim.Adam(
            self.generator.parameters(),
            lr=self.hparams.lr,
            betas=(self.hparams.beta1, self.hparams.beta2)
        )

        if self.regularization:
            optimizer_D = optim.Adam(
                self.regularization.D.parameters(),
                lr=self.hparams.lr,
                betas=(self.hparams.beta1, self.hparams.beta2)
            )
        else:
            optimizer_D = None

        return [optimizer_G, optimizer_D] if optimizer_D else [optimizer_G]
