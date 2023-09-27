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
import torch.optim as optim
import lightning.pytorch as pl
from typing import Dict, Optional, Sequence

from data.molecule import one_hot_encodings_to_tokens, tokens_to_selfies
from models.objective import SELFIESObjective
from models.fcnn import FCNN
from models.rnn import RNN
from models.regularization import Regularization


class MolGANModule(pl.LightningModule):
    """
    Molecule generation to optimize against an objective with optional source
    critic regularization.
    """

    def __init__(
        self,
        vocab: Dict[str, int],
        architecture: str,
        max_molecule_length: int = 109,
        alpha: float = 0.0,
        regularization: str = "gan_loss",
        num_dimensions: int = 128,
        num_layers: int = 3,
        embedding_layer_size: int = 128,
        dropout: float = 0.1,
        padding_token: str = "[pad]",
        lr: float = 0.0001,
        clip: Optional[float] = None,
        beta1: float = 0.5,
        beta2: float = 0.999,
        n_critic_per_generator: float = 1.0,
        c: float = 0.01,
        **kwargs
    ):
        """
        Args:
            vocab: vocabulary dict.
            architecture: GAN backbone to use. One of [`rnn`, `fcnn`].
            max_molecule_length: maximum molecule length. Default 109.
            alpha: source critic regularization weighting term. Default no
                source critic regularization.
            regularization: method of regularization. One of [`None`, `fid`,
                `gan_loss`, `importance_weighting`, `log_importance_weighting`,
                `wasserstein`, `em`].
            num_dimensions: number of dimensions of the RNN hidden states.
            num_layers: number of RNN layers.
            embedding_layer_size: size of the embedding layer.
            dropout: dropout parameter.
            use_bidirectional: whether to use a bidirectional RNN.
            padding_token: padding token in vocab. Default `[pad]`.
            lr: learning rate. Default 0.0001.
            clip: gradient clipping. Default no clipping.
            beta1: beta_1 parameter in Adam optimizer algorithm. Default 0.5.
            beta2: beta_2 parameter in Adam optimizer algorithm. Default 0.999.
            n_critic_per_generator: number of times to optimize the critic
                versus the generator. Default 1.0.
            c: weight clipping to enforce 1-Lipschitz condition on source
                critic for `wasserstein`/`em` regularization algorithms.
        """
        super().__init__()
        self.save_hyperparameters()
        self.hparams.alpha = min(max(alpha, 0.0), 1.0)
        self.automatic_optimization = False

        if self.hparams.architecture.lower() == "rnn":
            self.generator = RNN(
                cell_type="LSTM",
                out_dim=len(self.hparams.vocab.keys()),
                vocab=self.hparams.vocab,
                num_dimensions=self.hparams.num_dimensions,
                num_layers=self.hparams.num_layers,
                embedding_layer_size=self.hparams.embedding_layer_size,
                dropout=self.hparams.dropout,
                use_bidirectional=False,
                padding_token=self.hparams.padding_token,
                device=self.device
            )
        elif self.hparams.architecture.lower() == "fcnn":
            out_dim = self.hparams.max_molecule_length * len(
                self.hparams.vocab.keys()
            )
            self.generator = FCNN(
                in_dim=self.hparams.embedding_layer_size,
                out_dim=out_dim,
                hidden_dims=[2_048, 4_096],
                dropout=self.hparams.dropout
            )
        else:
            raise NotImplementedError(
                f"Backbone {self.hparams.architecture} not implemented."
            )

        self.regularization = Regularization(
            method=regularization,
            x_dim=self.hparams.max_molecule_length,
            vocab=self.hparams.vocab,
            c=self.hparams.c,
            use_rnn=bool(self.hparams.architecture.lower() == "rnn"),
            device=self.device
        )
        self.critic_loss = self.regularization.critic_loss
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

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """
        Generates a batch of molecules using the generator.
        Input:
            batch: input tokens with dimensions BN, where B is the batch size
                and N is the padded sequence length.
        Returns:
            Generated molecules with dimensions BN.
        """
        B, N = batch.size()
        if self.hparams.architecture.lower() == "rnn":
            X = torch.full(
                (B, 1),
                self.hparams.vocab[self.hparams.padding_token],
                dtype=torch.int
            )
            h = None
            state = torch.ones(B, 1, dtype=torch.int, device=self.device)
            molecules = torch.zeros_like(batch)
            for i in range(self.hparams.max_molecule_length):
                logits, h = self.generator(X, h)
                probs = torch.squeeze(torch.softmax(logits, dim=-1))
                X = torch.multinomial(probs, 1) * state
                state = (X > 1).to(torch.int)
                molecules[:, i] = torch.squeeze(X)
                if torch.sum(state) == 0:
                    break
        elif self.hparams.architecture.lower() == "fcnn":
            z = torch.randn((B, self.hparams.embedding_layer_size)).to(
                batch.device
            )
            molecules = self.generator(z).reshape(
                (B, N, len(self.hparams.vocab.keys()))
            )
            molecules = torch.argmax(molecules, dim=-1)
        return molecules

    @torch.no_grad()
    def sample(self, n: int) -> Sequence[str]:
        """
        Samples a batch of n molecules using the generator.
        Input:
            n: number of molecules to sample using the generator.
        Returns:
            List of n molecules as SELFIES strings generated by the network.
        """
        training_state = self.generator.training
        if training_state:
            self.generator.eval()
        if self.hparams.architecture.lower() == "rnn":
            dummy_batch = torch.zeros(
                (n, self.hparams.max_molecule_length), dtype=torch.int
            )
            molecules = self(dummy_batch)
        elif self.hparams.architecture.lower() == "fcnn":
            z = torch.randn((n, self.hparams.embedding_layer_size))
            molecules = self(z)
        if training_state:
            self.generator.train()
        return tokens_to_selfies(molecules, self.hparams.vocab)

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        """
        Training step for generative model.
        Input:
            batch: batch of input data for model training.
            batch_idx: batch index.
        Returns:
            None.
        """
        batch = one_hot_encodings_to_tokens(batch)
        B, N = batch.size()
        optimizer_G, optimizer_D = self.optimizers()
        generated = self(batch)

        if batch_idx % self.f_G == 0:
            self.toggle_optimizer(optimizer_G)

            train_obj = torch.mean(self.objective(generated))
            train_reg = self.regularization(batch, generated)
            loss_G = ((self.hparams.alpha - 1.0) * train_obj) + (
                self.hparams.alpha * train_reg
            )

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

        if batch_idx % self.f_D == 0 and self.hparams.alpha > 0.0:
            self.toggle_optimizer(optimizer_D)
            loss_D = self.critic_loss(batch, generated)

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
        batch = one_hot_encodings_to_tokens(batch)
        B, N = batch.size()
        optimizer_G, optimizer_D = self.optimizers()
        generated = self(batch)

        val_obj = torch.mean(self.objective(generated))
        val_reg = self.regularization(batch, generated)
        val_loss = ((self.hparams.alpha - 1.0) * val_obj) + (
            self.hparams.alpha * val_reg
        )

        self.log("val_loss", val_loss, prog_bar=False, sync_dist=True)
        self.log("val_obj", val_obj, prog_bar=True, sync_dist=True)
        self.log("val_reg", val_reg, prog_bar=True, sync_dist=True)

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
        if self.hparams.regularization in ["wasserstein", "em"]:
            optimizer_G = optim.RMSprop(
                self.generator.parameters(), lr=self.hparams.lr
            )
            optimizer_D = optim.RMSprop(
                self.regularization.f.parameters(), lr=self.hparams.lr
            )
            return [optimizer_G, optimizer_D]
        else:
            optimizer_G = optim.SGD(
                self.generator.parameters(),
                lr=self.hparams.lr
            )
            optimizer_D = optim.SGD(
                self.regularization.D.parameters(),
                lr=self.hparams.lr
            )
            return [optimizer_G, optimizer_D]
