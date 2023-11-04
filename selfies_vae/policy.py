"""
Bayesian optimization sampler policy for molecular generative adversarial
optimization.

Author(s):
    Yimeng Zeng @yimeng-zeng
    Michael Yao @michael-s-yao

Licensed under the MIT License. Copyright University of Pennsylvania 2023.
"""
import sys
import torch
import botorch
import selfies as sf
from pathlib import Path
from typing import Sequence, Union
from botorch.acquisition import qExpectedImprovement
from botorch.optim import optimize_acqf

sys.path.append(".")
from selfies_vae.data import SELFIESDataset
from selfies_vae.vae import InfoTransformerVAE
from selfies_vae.turbostate import TurboState


class BOPolicy:
    """Implements a Bayesian optimization policy for molecule generation."""

    def __init__(
        self,
        model: Union[Path, str],
        device: torch.device = torch.device("cpu"),
        num_restarts: int = 10,
        raw_samples: int = 512,
        **kwargs
    ):
        """
        Args:
            model: file path to trained transformer VAE state dict.
            device: device. Default CPU.
            num_restarts: number of starting points for multistart acquisition
                function optimization.
            raw_samples: number of samples for initialization.
        """
        self.model = model
        self.dataset = SELFIESDataset()
        self.device = device
        self.num_restarts = num_restarts
        self.raw_samples = raw_samples
        self.vae = InfoTransformerVAE(self.dataset).to(self.device)
        self.vae.load_state_dict(
            torch.load(self.model, map_location=self.device), strict=True
        )
        self.vae.eval()
        self.z_dim = self.vae.encoder_embedding_dim
        self.cache = {}
        self.state = TurboState(**kwargs)
        self.init_region_size = 6

    def __call__(
        self,
        model: botorch.models.model.Model,
        X: torch.Tensor,
        y: torch.Tensor,
        batch_size: int
    ) -> torch.Tensor:
        """
        Generate a set of candidates with the trust region via multi-start
        optimization.
        Input:
            model: a single-task variational GP model.
            X: input batch of encoded molecules.
            y: objective values of the input batch of encoded molecules.
            batch_size: number of candidates to return.
        """
        X_center = X[torch.argmax(y), :].clone()
        tr_lb = X_center - (self.init_region_size * self.state.length / 2)
        tr_ub = X_center + (self.init_region_size * self.state.length / 2)
        X_next, _ = optimize_acqf(
            qExpectedImprovement(model, torch.max(y), maximize=True),
            bounds=torch.stack([tr_lb, tr_ub]),
            q=batch_size,
            num_restarts=self.num_restarts,
            raw_samples=self.raw_samples,
        )
        return X_next

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
            z=z.reshape(-1, 2, self.z_dim // 2).to(
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
