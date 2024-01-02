"""
Main driver program for molecule generation-related training.

Author(s):
    Michael Yao

Licensed under the MIT License. Copyright University of Pennsylvania 2023.
"""
import argparse
import json
import math
import numpy as np
import pickle
import selfies as sf
import sys
import torch
import warnings
from pathlib import Path
from typing import Optional, Union
from botorch.exceptions.warnings import BadInitialCandidatesWarning
from botorch.utils.transforms import unnormalize

sys.path.append(".")
from molecules.data import SELFIESDataModule
from molecules.vae import InfoTransformerVAE
from molecules.policy import MoleculePolicy
from molecules.utils import MoleculeObjective
from models.fcnn import FCNN
from experiment.utility import seed_everything


def main(
    alpha: Optional[float] = None,
    num_generator_per_critic: int = 4,
    budget: int = 256,
    batch_size: int = 16,
    z_bound: float = 8.0,
    vae_ckpt: Union[Path, str] = "./molecules/checkpoints/vae.pt",
    surrogate_ckpt: Union[Path, str] = "./molecules/checkpoints/surrogate.pt",
    device: torch.device = torch.device("cpu"),
    savepath: Optional[Union[Path, str]] = None
) -> None:
    """
    Input:
        alpha: optional constant value for alpha.
        num_generator_per_critic: number of times to sample over the latent
            space before retraining the source critic. Default 4.
        budget: sampling budget.
        batch_size: batch size.
        z_bound: L-infinity sampling bound in the VAE latent space. Default 10.
        vae_ckpt: path to the trained molecule VAE model checkpoint file.
        surrogate_ckpt: path to the trained surrogate estimator checkpoint.
        device: device. Default CPU.
        savepath: optional path to save the optimization results to.
    Returns:
        None.
    """
    # Load the trained autoencoder.
    vae = InfoTransformerVAE().to(device)
    vae.load_state_dict(torch.load(vae_ckpt))
    vae.eval()
    z_dim = vae.encoder_embedding_dim

    # Load the surrogate objective function.
    with open("./molecules/hparams.json", "rb") as f:
        surrogate = FCNN(
            in_dim=z_dim,
            out_dim=1,
            hidden_dims=json.load(f)["hidden_dims"],
            dropout=0.0,
            final_activation=None,
            hidden_activation="ReLU",
            use_batch_norm=False
        )
    surrogate = surrogate.to(device)
    surrogate.load_state_dict(torch.load(surrogate_ckpt))
    surrogate.eval()

    # Define the oracle objective function.
    oracle = MoleculeObjective("logP")

    # Define the sampling bounds.
    bounds = torch.tensor(
        [[-z_bound] * z_dim, [z_bound] * z_dim], device=device
    )

    # Initialize the sampling policy.
    dm = SELFIESDataModule(num_workers=0)
    dm.prepare_data()
    dm.setup(None)
    ref_dataset = dm.test
    policy = MoleculePolicy(
        bounds,
        ref_dataset,
        surrogate,
        alpha=alpha,
        batch_size=batch_size,
        z_dim=z_dim,
        device=device
    )

    # Choose the initial set of observations.
    history = []
    z = unnormalize(
        torch.rand(batch_size, z_dim, device=device), bounds=bounds
    )
    z_ref = vae(policy.reference_sample(8 * batch_size))["z"]
    z_ref = z_ref.flatten(start_dim=(z_ref.ndim - 2))

    policy.fit_critic(z.detach(), z_ref.detach())
    alpha = policy.alpha(z_ref)
    if isinstance(alpha, torch.Tensor):
        history.append(alpha.item())
    else:
        history.append(alpha)
    y = (1.0 - alpha) * torch.squeeze(surrogate(z), dim=-1) - alpha * (
        policy.wasserstein(z_ref.detach(), z.detach())
    )
    y_gt = [
        oracle(sf.decoder(ref_dataset.decode(tok)))
        for tok in vae.sample(z=z.reshape(batch_size, 2, -1))
    ]
    samples = [
        (zz, yy, yy_gt)
        for zz, yy, yy_gt in zip(z, y, y_gt)
        if yy is not None and yy_gt is not None
    ]
    z = torch.stack([zz for zz, _, _ in samples]).to(device)
    y = torch.stack([yy for _, yy, _ in samples]).unsqueeze(dim=-1).to(device)
    y_gt = torch.unsqueeze(
        torch.tensor([yy_gt for _, _, yy_gt in samples]).to(device), dim=-1
    )

    # Generative adversarial Bayesian optimization.
    for step in range(math.ceil(budget / batch_size) - 1):
        policy.fit(z.detach(), y.detach())

        # Optimize and get new observations.
        new_z = policy(y)

        # Train the source critic.
        z_ref = vae(policy.reference_sample(8 * batch_size))["z"]
        z_ref = z_ref.flatten(start_dim=(z_ref.ndim - 2))
        if step % num_generator_per_critic == 0:
            policy.fit_critic(z.detach(), z_ref.detach())

        # Calculate the surrogate and objective values.
        alpha = policy.alpha(z_ref)
        if isinstance(alpha, torch.Tensor):
            history.append(alpha.item())
        else:
            history.append(alpha)
        new_y = (1.0 - alpha) * torch.squeeze(surrogate(new_z), dim=-1) - (
            alpha * policy.wasserstein(z_ref.detach(), new_z.detach())
        )
        new_gt = [
            oracle(sf.decoder(ref_dataset.decode(tok)))
            for tok in vae.sample(z=new_z.reshape(batch_size, 2, -1))
        ]
        samples = [
            (zz, yy, yy_gt)
            for zz, yy, yy_gt in zip(new_z, new_y, new_gt)
            if yy is not None and yy_gt is not None
        ]
        new_z = torch.stack([zz for zz, _, _ in samples]).to(device)
        new_y = torch.unsqueeze(
            torch.stack([yy for _, yy, _ in samples]).to(device), dim=-1
        )
        new_gt = torch.unsqueeze(
            torch.tensor([yy_gt for _, _, yy_gt in samples]).to(device), dim=-1
        )

        # Update training points.
        z = torch.cat((z, new_z), dim=0)
        y = torch.cat((y, new_y), dim=0)
        y_gt = torch.cat((y_gt, new_gt), dim=0)

        # Update progress.
        policy.save_current_state_dict()

    # Save optimization results.
    if savepath is not None:
        with open(savepath, mode="wb") as f:
            results = {
                "z": z.detach().cpu().numpy(),
                "y": y.detach().cpu().numpy(),
                "y_gt": y_gt.detach().cpu().numpy(),
                "alpha": np.array(history),
                "batch_size": batch_size,
                "budget": budget
            }
            pickle.dump(results, f)
    idx = torch.argmax(y)
    print(y[idx], y_gt[idx])


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=BadInitialCandidatesWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    parser = argparse.ArgumentParser(description="Molecule GABO Experiments")
    parser.add_argument(
        "--alpha", type=float, default=None, help="Optional constant alpha."
    )
    parser.add_argument(
        "--savepath", type=str, default=None, help="Path to save results to."
    )
    parser.add_argument(
        "--device", type=str, default="cpu", help="Device. Default CPU."
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed. Default 42."
    )
    args = parser.parse_args()

    seed_everything(args.seed)
    torch.set_default_dtype(torch.float64)
    main(
        alpha=args.alpha,
        device=torch.device(args.device),
        savepath=args.savepath
    )
