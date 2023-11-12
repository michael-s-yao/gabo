"""
Bayesian optimization sampler policy for molecular generative adversarial
optimization.

Author(s):
    Yimeng Zeng @yimeng-zeng
    Michael Yao @michael-s-yao

Licensed under the MIT License. Copyright University of Pennsylvania 2023.
"""
import numpy as np
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import botorch
import selfies as sf
from tqdm import tqdm
from typing import Optional, Sequence, Union
from botorch.acquisition import qExpectedImprovement
from botorch.optim import optimize_acqf

sys.path.append(".")
from models.fcnn import FCNN
from models.lipschitz import Lipschitz
from models.critic import WeightClipper
from selfies_vae.data import SELFIESDataset
from selfies_vae.vae import InfoTransformerVAE
from models.turbostate import TurboState


class BOPolicy:
    """Implements a Bayesian optimization policy for molecule generation."""

    def __init__(
        self,
        vae: InfoTransformerVAE,
        device: torch.device = torch.device("cpu"),
        num_restarts: int = 10,
        raw_samples: int = 512,
        **kwargs
    ):
        """
        Args:
            vae: trained transformer VAE autoencoder.
            device: device. Default CPU.
            num_restarts: number of starting points for multistart acquisition
                function optimization.
            raw_samples: number of samples for initialization.
        """
        self.dataset = SELFIESDataset()
        self.device = device
        self.num_restarts = num_restarts
        self.raw_samples = raw_samples
        self.vae = vae.to(self.device)
        self.cache = {}
        self.state = TurboState(**kwargs)
        self.init_region_size = 6

    def __call__(
        self,
        model: botorch.models.model.Model,
        z: torch.Tensor,
        y: torch.Tensor,
        batch_size: int
    ) -> torch.Tensor:
        """
        Generate a set of candidates with the trust region via multi-start
        optimization.
        Input:
            model: a single-task variational GP model.
            z: prior observations of encoded molecules.
            y: objective values of the prior observations of encoded molecules.
            batch_size: number of candidates to return.
        """
        z_center = z[torch.argmax(y), :].clone()
        tr_lb = z_center - (self.init_region_size * self.state.length / 2)
        tr_ub = z_center + (self.init_region_size * self.state.length / 2)
        z_next, _ = optimize_acqf(
            qExpectedImprovement(model, torch.max(y), maximize=True),
            bounds=torch.stack([tr_lb, tr_ub]),
            q=batch_size,
            num_restarts=self.num_restarts,
            raw_samples=self.raw_samples,
        )
        return z_next

    def update_state(self, y: torch.Tensor) -> None:
        """
        Updates the state internal variables given objective values y.
        Input:
            y: input objective values.
        Returns:
           None.
        """
        return self.state.update(y)

    def decode(self, z: torch.Tensor) -> Sequence[str]:
        """
        Decodes a tensor of VAE latent space represntations to SMILES molecule
        representations.
        Input:
            z: a tensor of the encoded molecules in the VAE latent space.
        Returns:
            A list of decoded SMILES molecule representations.
        """
        sample = self.vae.sample(
            z=z.reshape(-1, 2, self.vae.encoder_embedding_dim // 2).to(
                device=self.device, dtype=self.vae.dtype
            )
        )
        selfies = [
            self.dataset.decode(sample[i]) for i in range(sample.size(dim=-2))
        ]
        return [sf.decoder(mol) for mol in selfies]

    def encode(self, smiles: Sequence[str]) -> torch.Tensor:
        """
        Encodes a list of SMILES representations to the VAE latent space.
        Input:
            smiles: a list of SMILES molecule representations to encode.
        Returns:
            A tensor of the encoded molecules in the VAE latent space.
        """
        X_list = []
        for smi in smiles:
            selfie = self.cache.get(smi, None)
            if selfie is None:
                selfie = sf.encoder(smi)
                self.cache[smi] = selfie
            tokenized_selfie = self.dataset.tokenize_selfies([selfie])[0]
            encoded_selfie = torch.unsqueeze(
                self.dataset.encode(tokenized_selfie), dim=0
            )
            X_list.append(encoded_selfie)
        X = self.dataset.collate_fn(X_list).to(self.device)
        return self.vae(X)["z"].reshape(-1, self.vae.encoder_embedding_dim)

    @property
    def restart_triggered(self) -> bool:
        """
        Returns whether a restart has been triggered during the optimization.
        Input:
            None.
        Returns:
            Whether a restart has been triggered during the optimization.
        """
        return self.state.restart_triggered


class BOAdversarialPolicy(BOPolicy):
    def __init__(
        self,
        vae: InfoTransformerVAE,
        alpha: Union[float, str],
        surrogate: Optional[nn.Module] = None,
        device: torch.device = torch.device("cpu"),
        num_restarts: int = 10,
        raw_samples: int = 512,
        critic_lr: float = 0.001,
        critic_max_steps: int = 50,
        critic_batch_size: int = 128,
        critic_c: float = 0.1,
        verbose: bool = True,
        seed: int = 42,
        **kwargs
    ):
        """
        Args:
            vae: trained transformer VAE autoencoder.
            alpha: a float between 0 and 1, or `Lipschitz` for our method.
            surrogate: surrogate function for objective estimation. Only
                required if the alpha argument is `Lipschitz`.
            device: device. Default CPU.
            num_restarts: number of starting points for multistart acquisition
                function optimization.
            raw_samples: number of samples for initialization.
            critic_lr: learning rate for source critic training.
            critic_max_steps: maximum number of weight updates per BO step.
            critic_batch_size: batch size for source critic training.
            critic_c: source critic weight clipping parameter.
            verbose: whether to print verbose outputs to `stdout`.
            seed: random seed. Default 42.
        """
        super().__init__(vae, device, num_restarts, raw_samples, **kwargs)
        self.alpha_ = alpha
        if self.alpha_.replace(".", "", 1).isnumeric():
            self.alpha_ = float(self.alpha_)
        self.surrogate = surrogate
        self.critic_lr = critic_lr
        self.critic_max_steps = critic_max_steps
        self.critic_batch_size = critic_batch_size
        self.critic_c = critic_c
        self.verbose = verbose
        self.seed = seed
        self.rng = np.random.RandomState(seed=self.seed)
        self.critic = FCNN(
            in_dim=self.vae.encoder_embedding_dim,
            out_dim=1,
            hidden_dims=[1024, 256, 64],
            dropout=0.0,
            final_activation=None,
            hidden_activation="ReLU"
        )
        self.clipper = WeightClipper(c=self.critic_c)
        self.critic_optimizer = self.configure_critic_optimizers()
        if isinstance(self.alpha_, str):
            self.L = Lipschitz(self.surrogate, mode="local", p=2)
            self.K = Lipschitz(self.critic, mode="global")

    def update_state(
        self, y: torch.Tensor, z: torch.Tensor, ref_dataset: SELFIESDataset
    ) -> torch.Tensor:
        """
        Updates the state internal variables given objective values y.
        Input:
            y: input objective values.
            z: input molecules in the latent space of the encoder
                corresponding to the input objective values.
            ref_dataset: reference dataset of in-domain molecules.
        Returns:
            A tensor of the penalized objective values.
        """
        alpha = self.alpha(z)
        zref = self.reference_sample(ref_dataset, z.size(dim=0))
        penalized_objective = ((1 - alpha) * torch.squeeze(y, dim=-1)) - (
            alpha * torch.squeeze(
                torch.maximum(
                    torch.mean(self.critic(zref)) - self.critic(z),
                    torch.zeros_like(y)
                ),
                dim=-1
            )
        )
        penalized_objective = torch.unsqueeze(penalized_objective, dim=-1)
        self.state.update(penalized_objective)
        return penalized_objective

    def alpha(self, Zq: torch.Tensor) -> Union[float, torch.Tensor]:
        """
        Calculates the value of alpha for regularization weighting.
        Input:
            Xq: generated molecules in the latent space of the encoder.
        Returns:
            The value(s) of alpha for regularization weighting.
        """
        if isinstance(self.alpha_, float):
            return self.alpha_
        L = self.L(Zq)
        return torch.from_numpy(L / (L + self.K())).to(Zq)

    def update_critic(
        self,
        model: botorch.models.model.Model,
        z: torch.Tensor,
        y: torch.Tensor,
        ref_dataset: SELFIESDataset
    ) -> None:
        """
        Trains the source critic according to the allocated training budget.
        Input:
            model: a single-task variational GP model.
            z: prior observations of encoded molecules.
            y: objective values of the prior observations of encoded molecules.
            ref_dataset: reference dataset of in-domain molecules.
        Returns:
            None.
        """
        if isinstance(self.alpha_, float) and self.alpha_ == 0.0:
            return
        for _ in tqdm(
            range(self.critic_max_steps),
            desc="Training Source Critic",
            leave=False,
            disable=(not self.verbose)
        ):
            Zp = self.reference_sample(ref_dataset, self.critic_batch_size)
            Zq = self(model, z, y, batch_size=self.critic_batch_size)
            self.critic.zero_grad()
            negWd = torch.mean(self.critic(Zq)) - torch.mean(self.critic(Zp))
            negWd.backward()
            self.critic_optimizer.step()
            self.clipper(self.critic)
        return

    def configure_critic_optimizers(self) -> optim.Optimizer:
        """
        Returns the optimizer for the source critic.
        Input:
            None.
        Returns:
            The optimizer for the source critic.
        """
        return optim.SGD(self.critic.parameters(), lr=self.critic_lr)

    def reference_sample(
        self, ref_dataset: SELFIESDataset, num: int
    ) -> torch.Tensor:
        """
        Samples a batch of random real molecules from a reference dataset.
        Input:
            ref_dataset: reference dataset of real molecules.
            num: number of molecules to sample from the reference dataset.
        Returns:
            A batch of real molecules from the reference dataset encoded in
            the VAE latent space.
        """
        idxs = self.rng.choice(
            len(ref_dataset), min(num, len(ref_dataset)), replace=False
        )
        Zp = [
            self.vae(ref_dataset[int(i)].unsqueeze(dim=0).to(self.device))
            for i in idxs
        ]
        return torch.cat(
            [
                out["z"].reshape(-1, self.vae.encoder_embedding_dim)
                for out in Zp
            ],
            dim=0
        )
