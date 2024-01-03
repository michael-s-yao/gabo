"""
Implements a VAE model with transformer encoder and decoder layers.

Author(s):
    Yimeng Zeng @yimeng-zeng
    Michael Yao @michael-s-yao

Citation(s):
    [1] Maus NT, Jones HT, Moore JS, Kusner MJ, Bradshaw J, Gardner JR.
        Local latent space Bayesian optimization over structured inputs.
        Proc NeurIPS. (2022). https://doi.org/10.48550/arXiv.2201.11872

Licensed under the MIT License. Copyright University of Pennsylvania 2023.
"""
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
from collections import namedtuple
from data.molecules.selfies import SELFIESDataModule, SELFIESDataset
from pytorch_lightning.callbacks import ModelCheckpoint, RichProgressBar
from typing import Any, Dict, Optional, Tuple, Union

sys.path.append(".")
from models.pe import PositionalEncoding


class InfoTransformerVAE(nn.Module):
    def __init__(
        self,
        vocab2idx: Dict[str, int],
        start: str,
        stop: str,
        bottleneck_size: int = 2,
        model_dim: int = 128,
        is_autoencoder: bool = False,
        kl_factor: float = 0.1,
        min_posterior_std: float = 1e-4,
        max_string_length: int = 256,
        encoder_nhead: int = 8,
        encoder_dim_feedforward: int = 512,
        encoder_dropout: float = 0.1,
        encoder_num_layers: int = 6,
        decoder_nhead: int = 8,
        decoder_dim_feedforward: int = 256,
        decoder_dropout: float = 0.1,
        decoder_num_layers: int = 6,
        max_len: int = 5_000,
        **kwargs
    ):
        """
        Args:
            vocab2idx: a mapping of SELFIES tokens to integer values.
            start: the start token from the dictionary.
            stop: the stop token from the dictionary.
            bottleneck_size: size of the model bottleneck. Default 2.
            model_dim: model dimensions. Default 128.
            is_autoencoder: whether to treat the VAE model as an autoencoder.
            kl_factor: relative weighting of the KL Divergence term in the loss
                function.
            min_posterior_std: minimum value for the standard deviation of the
                posterior distribution.
            max_string_length: maximum string length of a generated molecule.
            encoder_nhead: number of encoder attention heads. Default 8.
            encoder_dim_feedforward: number of feedforward dimensions for the
                encoder. Default 512.
            encoder_dropout: encoder dropout probability. Default 0.1.
            encoder_num_layers: number of layers for the encoder. Default 6.
            decoder_nhead: number of decoder attention heads. Default 8.
            decoder_dim_feedforward: number of feedforward dimensions for the
                decoder. Default 256.
            decoder_dropout: decoder dropout probability. Default 0.1.
            decoder_num_layers: number of layers for the decoder. Default 6.
            max_len: maximum input length to use for positional encoding.
                Default 5,000.
        """
        super().__init__()
        self.hparams = locals()
        self.save_hyperparameters()
        self.vocab_size = len(self.hparams.vocab2idx.keys())
        self.latent_size = self.hparams.model_dim

        self.encoder_embedding_dim = 2 * self.hparams.model_dim

        self.encoder_token_embedding = nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.encoder_embedding_dim
        )
        self.encoder_position_encoding = PositionalEncoding(
            model_dim=self.encoder_embedding_dim,
            dropout=self.hparams.encoder_dropout,
            max_len=self.hparams.max_len
        )
        self.decoder_token_embedding = nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.hparams.model_dim
        )
        self.decoder_position_encoding = PositionalEncoding(
            model_dim=self.hparams.model_dim,
            dropout=self.hparams.decoder_dropout,
            max_len=self.hparams.max_len
        )
        self.decoder_token_unembedding = nn.Parameter(
            torch.randn(self.hparams.model_dim, self.vocab_size)
        )
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.encoder_embedding_dim,
                nhead=self.hparams.encoder_nhead,
                dim_feedforward=self.hparams.encoder_dim_feedforward,
                dropout=self.hparams.encoder_dropout,
                activation="relu"
            ),
            num_layers=self.hparams.encoder_num_layers
        )
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=self.hparams.model_dim,
                nhead=self.hparams.decoder_nhead,
                dim_feedforward=self.hparams.decoder_dim_feedforward,
                dropout=self.hparams.decoder_dropout,
                activation="relu"
            ),
            num_layers=self.hparams.decoder_num_layers
        )

    def sample_prior(self, n: int) -> torch.Tensor:
        """
        Sample n datums from the prior distribution, which is just the
        standard normal distribution.
        Input:
            n: number of datums to sample.
        Returns:
            A tensor of n datums from the prior distribution.
        """
        prior = torch.randn(
            n, self.hparams.bottleneck_size, self.hparams.model_dim
        )
        return prior.to(self.device)

    def sample_posterior(
        self, mu: torch.Tensor, sigma: torch.Tensor, n: Optional[int] = None
    ) -> torch.Tensor:
        """
        Sample n datums from the posterior distribution, which is a
        multivariate normal distribution with mean mu and standard deviation
        sigma.
        Input:
            mu: the mean of the multivariate normal distribution.
            sigma: the standard deviation of the multivariate normal
                distribution.
            n: number of datums to sample.
        Returns:
            A tensor of n datums from the posterior distribution.
        """
        if n is not None:
            mu = torch.unsqueeze(mu, dim=0).expand(n, -1, -1, -1)
        return mu + (torch.randn_like(mu) * sigma)

    def generate_pad_mask(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Generates a mask that tells the encoder to ignore all but the first
        stop token.
        Input:
            tokens: token representations of molecules.
        Returns:
            The corresponding mask telling the encoder to ignore all but the
            first stop token.
        """
        mask = tokens == self.hparams.vocab2idx[self.hparams.stop]
        idxs = torch.argmax(mask.float(), dim=-1)
        mask[torch.arange(0, tokens.shape[0]), idxs] = False
        return mask.to(torch.bool)

    def encode(
        self, tokens: torch.Tensor, as_probs: bool = False
    ) -> Tuple[torch.Tensor]:
        """
        Encodes an input tensor of tokens into the VAE latent space.
        Input:
            tokens: an input tensor of tokens.
            as_probs: whether the input tensor of tokens are probabilities.
        Returns:
            The mean and standard deviation of the encoded tokens in the VAE
            latent space.
        """
        if as_probs:
            embed = tokens @ self.encoder_token_embedding.weight
        else:
            embed = self.encoder_token_embedding(tokens)
        embed = self.encoder_position_encoding(embed)
        pad_mask = self.generate_pad_mask(tokens)
        print(embed.size())
        encoding = self.encoder(embed, src_key_padding_mask=pad_mask)
        mu = encoding[..., :self.hparams.model_dim]
        sigma = self.hparams.min_posterior_std + F.softplus(
            encoding[..., self.hparams.model_dim:]
        )
        mu = mu[:, :self.hparams.bottleneck_size, :]
        sigma = sigma[:, :self.hparams.bottleneck_size, :]
        return mu, sigma

    def decode(
        self, z: torch.Tensor, tokens: torch.Tensor, as_probs: bool = False
    ) -> torch.Tensor:
        """
        Input:
            z: input points in the VAE latent space.
            tokens: an input tensor of tokens.
            as_probs: whether the input tensor of tokens are probabilities.
        Returns:
            The decoded logits corresponding to molecule token representations.
        """
        if as_probs:
            embed = tokens[:, :-1] @ self.decoder_token_embedding.weight
        else:
            embed = self.decoder_token_embedding(tokens[:, :-1])

        embed = torch.cat(
            [
                torch.full(
                    (embed.shape[0], 1, embed.shape[-1]),
                    fill_value=self.hparams.vocab2idx[self.hparams.start],
                    device=self.device
                ),
                embed
            ],
            dim=1
        )
        embed = self.decoder_position_encoding(embed)

        # TODO: Mask out all stop tokens but the first?
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(
            embed.shape[1]
        )
        decoding = self.decoder(
            tgt=embed, memory=z, tgt_mask=tgt_mask.to(
                device=self.device, dtype=torch.bool
            )
        )
        logits = decoding @ self.decoder_token_unembedding

        return logits

    @torch.no_grad()
    def sample(
        self,
        n: Optional[int] = -1,
        z: Optional[torch.Tensor] = None,
        differentiable: bool = False,
        return_logits: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor]]:
        """
        Samples and decodes n datums from the VAE latent space distribution.
        Input:
            n: number of points to sample.
            z: optional specified latent space points to decode.
            differentiable: whether function outputs should be differentiable.
            return_logits: whether to return the decoder logits in addition
                to the sampled molecules. Default False.
        Returns:
            sample: sampled decoded molecules.
            logits: returned if `return_logits` is True.
        """
        model_state = self.training
        self.eval()

        if z is None:
            z = self.sample_prior(n)
        else:
            n = z.shape[0]

        tokens = torch.full(
            (n, 1),
            fill_value=self.hparams.vocab2idx[self.hparams.start],
            dtype=torch.long,
            device=self.device
        )
        for _ in range(self.hparams.max_string_length):
            tgt = self.decoder_token_embedding(tokens)
            tgt = self.decoder_position_encoding(tgt)
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(
                tokens.shape[-1]
            )
            tgt_mask = tgt_mask.to(dtype=torch.bool, device=self.device)

            decoding = self.decoder(tgt=tgt, memory=z, tgt_mask=tgt_mask)
            logits = decoding @ self.decoder_token_unembedding
            sample, randoms = self.gumbel_softmax(
                logits, dim=-1, hard=True, return_randoms=True
            )

            tokens = torch.cat(
                [tokens, sample[:, -1, :].argmax(dim=-1)[:, None]], dim=-1
            )

            # Check if all molecules have a stop token in them.
            all_stop = torch.all(
                torch.sum(
                    tokens == self.hparams.vocab2idx[self.hparams.stop], dim=-1
                ) > 0
            )
            if all_stop.item():
                break

        self.train(model_state)
        if not differentiable:
            sample = tokens
        if return_logits:
            return sample, logits
        return sample

    def forward(self, tokens: torch.Tensor) -> Dict[str, Any]:
        """
        Forward propagation through the transformer VAE.
        Input:
            tokens: an input batch of tokens.
        Returns:
            A dictionary contining the molecule reconstruction results.
        """
        mu, sigma = self.encode(tokens)
        if self.hparams.is_autoencoder:
            z = mu
        else:
            z = self.sample_posterior(mu, sigma)
        logits = self.decode(z, tokens)
        recon_loss = torch.mean(
            F.cross_entropy(logits.permute(0, 2, 1), tokens, reduction="none")
        )
        sigma2 = sigma.pow(2)
        kldiv = torch.mean(0.5 * (mu.pow(2) + sigma2 - sigma2.log() - 1))
        loss = recon_loss.clone()
        if self.hparams.kl_factor != 0:
            loss = loss + (self.hparams.kl_factor * kldiv)
        loss = loss
        return {
            "loss": loss,
            "z": z,
            "recon_loss": recon_loss,
            "kldiv": kldiv,
            "recon_token_acc": torch.mean(
                (logits.argmax(dim=-1) == tokens.float()).to(torch.float)
            ),
            "recon_string_acc": torch.mean(
                torch.all(logits.argmax(dim=-1) == tokens, dim=1).float(),
                dim=0
            ),
            "sigma_mean": torch.mean(sigma),
        }

    def save_hyperparameters(self) -> None:
        """
        Save input hyperparameters of the sampling policy.
        Input:
            None.
        Returns:
            None.
        """
        fields = set(self.hparams.keys())
        fields.discard("self"), fields.discard("__class__")
        self.hparams.pop("self", None), self.hparams.pop("__class__", None)
        self.hparams = namedtuple("hparams", fields)(**self.hparams)
        for hparam in self.hparams:
            if isinstance(hparam, (nn.Module, torch.Tensor)):
                hparam = hparam.to(self.hparams.device)

    def gumbel_softmax(
        self,
        logits: torch.Tensor,
        tau: float = 1,
        hard: bool = False,
        dim: int = -1,
        return_randoms: bool = False,
        randoms: Optional[torch.Tensor] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor]]:
        """
        Samples the Gumbel-Softmax distribution and optionally discretizes.
        Input:
            logits: input logits.
            tau: non-negative scalar temperature. Default 1.
            hard: whether to return the samples as discretized one-hot vectors.
            dim: the dimension along whcih softmax will be computed.
            return_randoms: whther to return the random values used for
                calculating the Gumbel-Softmax values.
            randoms: the optional values to use for calculating the
                Gumbel-Softmax values.
        Returns:
            ret: samples from the Gumbel-Softmax distribution.
            randoms: returned if `return_randoms` is True. The values used for
                calculating the Gumbel-Softmax values.
        Citation:
            [1] https://pytorch.org/docs/stable/generated/torch.nn.functional.
                gumbel_softmax.html#torch.nn.functional.gumbel_softmax
        """
        if randoms is None:
            # ~Gumbel(0, 1).
            randoms = -1.0 * torch.empty_like(
                logits, memory_format=torch.legacy_contiguous_format
            )
            randoms = randoms.exponential_().log()
        # ~Gumbel(logits, tau)
        gumbels = (logits + randoms) / tau
        y_soft = gumbels.softmax(dim)

        if hard:
            # Straight through.
            index = y_soft.max(dim, keepdim=True)[1]
            y_hard = torch.zeros_like(
                logits, memory_format=torch.legacy_contiguous_format
            )
            y_hard = y_hard.scatter_(dim, index, 1.0)
            ret = y_hard - y_soft.detach() + y_soft
        else:
            # Reparametrization trick.
            ret = y_soft

        return ret, randoms if return_randoms else ret


class VAEModule(pl.LightningModule):
    def __init__(
        self,
        dataset: SELFIESDataset,
        bottleneck_size: int = 2,
        model_dim: int = 128,
        is_autoencoder: bool = False,
        kl_factor: float = 0.1,
        min_posterior_std: float = 1e-4,
        encoder_nhead: int = 8,
        encoder_dim_feedforward: int = 512,
        encoder_dropout: float = 0.1,
        encoder_num_layers: int = 6,
        encoder_lr: float = 1e-3,
        encoder_warmup_steps: int = 100,
        aggressive_steps: int = 5,
        decoder_nhead: int = 8,
        decoder_dim_feedforward: int = 256,
        decoder_dropout: float = 0.1,
        decoder_num_layers: int = 6,
        decoder_lr: float = 1e-3,
        decoder_warmup_steps: int = 100,
        num_sample_per_epoch: int = 100
    ):
        """
        Args:
            dataset: dataset to use for model training.
            bottleneck_size: size of the model bottleneck. Default 2.
            model_dim: model dimensions. Default 128.
            is_autoencoder: whether to treat the VAE model as an autoencoder.
            kl_factor: relative weighting of the KL Divergence term in the loss
                function.
            min_posterior_std: minimum value for the standard deviation of the
                posterior distribution.
            encoder_nhead: number of encoder attention heads. Default 8.
            encoder_dim_feedforward: number of feedforward dimensions for the
                encoder. Default 512.
            encoder_dropout: encoder dropout probability. Default 0.1.
            encoder_num_layers: number of layers for the encoder. Default 6.
            encoder_lr: learning rate for the encoder. Default 0.001.
            encoder_warmup_steps: number of warmup steps for the encoder.
                Default 100.
            aggressive_steps: number of aggressive training steps. Default 5.
            decoder_nhead: number of decoder attention heads. Default 8.
            decoder_dim_feedforward: number of feedforward dimensions for the
                decoder. Default 256.
            decoder_dropout: decoder dropout probability. Default 0.1.
            decoder_num_layers: number of layers for the decoder. Default 6.
            decoder_lr: learning rate for the decoder. Default 0.001.
            decoder_warmup_steps: number of warmup steps for the decoder.
                Default 100.
            num_sample_per_epoch: number of sample molecules to generate at the
                end of each training epoch. Default 100.
        """
        super().__init__()
        self.save_hyperparameters(ignore="dataset")
        self.model = InfoTransformerVAE(dataset=dataset, **self.hparams)

    def training_step(
        self, batch: torch.Tensor, batch_idx: int
    ) -> Dict[str, Any]:
        """
        Implements the training step.
        Input:
            batch: batch of input molecule representations.
            batch_idx: index of the batch in the dataset.
        Returns:
            A dictionary of the computed result values.
        """
        outputs = {
            key: (val.detach() if key != "loss" else val)
            for key, val in self.model(batch).items()
        }
        for key, val in outputs.items():
            is_loss = key == "loss"
            if not is_loss:
                self.log(
                    key,
                    val,
                    on_step=(not is_loss),
                    on_epoch=is_loss,
                    prog_bar=(not is_loss),
                    logger=is_loss
                )
            self.log(
                f"train/{key}",
                val,
                on_step=is_loss,
                on_epoch=(not is_loss),
                prog_bar=is_loss,
                logger=(not is_loss)
            )
        return outputs

    def validation_step(self, batch, batch_idx):
        """
        Implements the validation step.
        Input:
            batch: batch of input molecule representations.
            batch_idx: index of the batch in the dataset.
        Returns:
            A dictionary of the computed result values.
        """
        outputs = self.model(batch)
        for key, val in outputs.items():
            self.log(
                f"val/{key}",
                val,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                logger=True,
                sync_dist=True
            )

    def on_train_epoch_end(self) -> None:
        """
        Samples from the latent space at the end of every training epoch.
        Input:
            None.
        Returns:
            None.
        """
        if self.trainer.is_global_zero:
            with torch.no_grad():
                samples = self.model.sample(self.hparams.num_sample_per_epoch)
                samples = list(map(self.model.dataset.decode, samples))

            with open(
                os.path.join(
                    self.trainer.logger.log_dir,
                    f"samples-epoch={self.current_epoch}.txt"
                ),
                "wt"
            ) as f:
                _ = [print(sample, file=f) for sample in samples]
        return

    def configure_optimizers(self) -> Dict[str, Any]:
        """
        Configure optimizers and LR schedulers.
        Input:
            None.
        Returns:
            A dictionary with the relevant optimizer and LR scheduler.
        """
        encoder_params, decoder_params = [], []
        for name, param in self.named_parameters():
            if param.requires_grad:
                if "encoder" in name:
                    encoder_params.append(param)
                elif "decoder" in name:
                    decoder_params.append(param)
                else:
                    raise ValueError(f"Unknown parameter {name}")

        optimizer = optim.Adam([
            {
                "params": encoder_params,
                "lr": self.hparams.encoder_lr
            },
            {
                "params": decoder_params,
                "lr": self.hparams.decoder_lr
            }
        ])
        lr_scheduler = optim.lr_scheduler.LambdaLR(
            optimizer, [self._encoder_lr_sched, self._decoder_lr_sched]
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step",
                "frequency": 1
            }
        }

    def _encoder_lr_sched(self, step: int) -> float:
        """
        Simple linear warmup LR scheduler for the encoder.
        Input:
            step: optimization step.
        Returns:
            LR weighting factor for the encoder.
        """
        return min(step / self.hparams.encoder_warmup_steps, 1.0)

    def _decoder_lr_sched(self, step: int) -> float:
        """
        Decoder LR scheduler.
        Input:
            step: optimization step.
        Returns:
            LR weighting factor for the decoder.
        """
        if step < self.hparams.encoder_warmup_steps or (
            (step - self.hparams.encoder_warmup_steps + 1) %
            self.hparams.aggressive_steps != 0
        ):
            return 0.0
        return min(
            (step - self.hparams.encoder_warmup_steps) / (
                self.hparams.decoder_warmup_steps *
                self.hparams.aggressive_steps
            ),
            1.0
        )


def fit() -> None:
    """
    Trains a transformer VAE on the SELFIES molecule dataset.
    Input:
        None.
    Returns:
        None.
    """
    datamodule = SELFIESDataModule()
    checkpath = None
    model = VAEModule(dataset=datamodule.train)
    logger = pl.loggers.TensorBoardLogger(
        save_dir="lightning_logs", name=type(model).__name__ + "_enc_masked"
    )
    check = ModelCheckpoint(
        every_n_epochs=10,
        save_top_k=-1,
        save_last=True
    )
    trainer = pl.Trainer(
        gpus=-1,
        strategy=pl.plugins.DDPPlugin(find_unused_parameters=False),
        logger=logger,
        callbacks=[check, RichProgressBar()],
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
        detect_anomaly=True,
        max_epochs=1_000,
    )
    trainer.fit(model, datamodule=datamodule, ckpt_path=checkpath)
