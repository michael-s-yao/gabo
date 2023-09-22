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
        c: float = 0.1,
        **kwargs
    ):
        """
        Args:
            vocab: vocabulary dict.
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

        self.regularization = Regularization(
            method=regularization,
            vocab=self.hparams.vocab,
            c=self.hparams.c,
            use_rnn=True,
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
        B, _ = batch.size()
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
        dummy_batch = torch.zeros((
            n, self.max_molecule_length, len(self.hparams.vocab.keys())
        ))
        molecules = self(dummy_batch)
        if training_state:
            self.generator.train()
        return tokens_to_selfies(molecules)

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
            optimizer_G = optim.Adam(
                self.generator.parameters(),
                lr=self.hparams.lr,
                betas=(self.hparams.beta1, self.hparams.beta2)
            )
            optimizer_D = optim.Adam(
                self.regularization.D.parameters(),
                lr=self.hparams.lr,
                betas=(self.hparams.beta1, self.hparams.beta2)
            )
            return [optimizer_G, optimizer_D]
