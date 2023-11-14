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
import torch.nn as nn
import botorch
import selfies as sf
from pathlib import Path
from tqdm import tqdm
from typing import Optional, Sequence, Union

sys.path.append(".")
from selfies_vae.data import SELFIESDataset
from selfies_vae.vae import InfoTransformerVAE
from models.policy import BOAdversarialPolicy


class SELFIESAdversarialPolicy(BOAdversarialPolicy):
    def __init__(
        self,
        ref_dataset: SELFIESDataset,
        vae: InfoTransformerVAE,
        alpha: Union[float, str],
        surrogate: Optional[nn.Module] = None,
        device: torch.device = torch.device("cpu"),
        num_restarts: int = 10,
        raw_samples: int = 512,
        critic_config: Union[Path, str] = "./selfies_vae/critic_config.json",
        verbose: bool = True,
        seed: int = 42,
        **kwargs
    ):
        """
        Args:
            ref_dataset: a reference dataset of real molecules.
            vae: trained transformer VAE autoencoder.
            alpha: a float between 0 and 1, or `Lipschitz` for our method.
            surrogate: surrogate function for objective estimation. Only
                required if the alpha argument is `Lipschitz`.
            device: device. Default CPU.
            num_restarts: number of starting points for multistart acquisition
                function optimization.
            raw_samples: number of samples for initialization.
            critic_config: JSON file with source critic hyperparameters.
            verbose: whether to print verbose outputs to `stdout`.
            seed: random seed. Default 42.
        """
        super().__init__(
            maximize=True,
            ref_dataset=ref_dataset,
            autoencoder=vae,
            alpha=alpha,
            surrogate=surrogate,
            device=device,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
            critic_config=critic_config,
            verbose=verbose,
            seed=seed
        )
        self.cache = {}
        self.z_dim = self.autoencoder.encoder_embedding_dim

    def reference_sample(self, num: int) -> torch.Tensor:
        """
        Samples a batch of random real molecules from a reference dataset.
        Input:
            num: number of molecules to sample from the reference dataset.
        Returns:
            A batch of real molecules from the reference dataset encoded in
            the VAE latent space.
        """
        idxs = self.rng.choice(len(self.ref_dataset), num, replace=False)
        Zp = [
            self.autoencoder(
                self.ref_dataset[int(i)].unsqueeze(dim=0).to(self.device)
            )
            for i in idxs
        ]
        return torch.cat(
            [
                out["z"].reshape(-1, self.autoencoder.encoder_embedding_dim)
                for out in Zp
            ],
            dim=0
        )

    def decode(self, z: torch.Tensor) -> Sequence[str]:
        """
        Decodes a tensor of VAE latent space represntations to SMILES molecule
        representations.
        Input:
            z: a tensor of the encoded molecules in the VAE latent space.
        Returns:
            A list of decoded SMILES molecule representations.
        """
        sample = self.autoencoder.sample(
            z=z.reshape(-1, 2, self.autoencoder.encoder_embedding_dim // 2).to(
                device=self.device, dtype=self.autoencoder.dtype
            )
        )
        selfies = [
            self.ref_dataset.decode(sample[i])
            for i in range(sample.size(dim=-2))
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
            tokenized_selfie = self.ref_dataset.tokenize_selfies([selfie])[0]
            encoded_selfie = torch.unsqueeze(
                self.ref_dataset.encode(tokenized_selfie), dim=0
            )
            X_list.append(encoded_selfie)
        X = self.ref_dataset.collate_fn(X_list).to(self.device)
        return self.autoencoder(X)["z"].reshape(
            -1, self.autoencoder.encoder_embedding_dim
        )

    def update_critic(
        self,
        model: botorch.models.model.Model,
        z: torch.Tensor,
        y: torch.Tensor
    ) -> None:
        """
        Trains the source critic according to the allocated training budget.
        Input:
            model: a single-task variational GP model.
            z: prior observations of encoded molecules.
            y: objective values of the prior observations of encoded molecules.
        Returns:
            None.
        """
        if isinstance(self.alpha_, float) and self.alpha_ == 0.0:
            return
        with tqdm(
            range(self.critic_config["max_steps"]),
            desc="Training Source Critic",
            leave=False,
            disable=(not self.verbose)
        ) as pbar:
            for _ in pbar:
                Zp = self.reference_sample(self.critic_config["batch_size"])
                Zq = self(model, z, y, self.critic_config["batch_size"])
                self.critic.zero_grad()
                negWd = torch.mean(self.critic(Zq)) - torch.mean(
                    self.critic(Zp)
                )
                negWd.backward()
                self.critic_optimizer.step()
                self.clipper(self.critic)
                pbar.set_postfix(Wd=-negWd.item())
        return
