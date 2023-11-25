"""
Main driver program for MNIST-related training.

Author(s):
    Michael Yao

Licensed under the MIT License. Copyright University of Pennsylvania 2023.
"""
import argparse
import math
import numpy as np
import pickle
import sys
import torch
import warnings
import torchvision as thv
from pathlib import Path
from typing import Optional, Union
from botorch.exceptions.warnings import BadInitialCandidatesWarning
from botorch.utils.transforms import unnormalize

sys.path.append(".")
from mnist.vae import VAE
from mnist.energy import Surrogate
from mnist.policy import MNISTPolicy
from experiment.utility import seed_everything


def main(
    alpha: Optional[float] = None,
    num_generator_per_critic: int = 4,
    budget: int = 512,
    batch_size: int = 16,
    z_bound: float = 5.0,
    root: Union[Path, str] = "./mnist/data",
    vae_ckpt: Union[Path, str] = "./mnist/checkpoints/mnist_vae.pt",
    energy_ckpt: Union[Path, str] = "./mnist/checkpoints/mnist_surrogate.pt",
    device: torch.device = torch.device("cpu"),
    savepath: Optional[Union[Path, str]] = None,
    seed: int = 42
) -> None:
    """
    Input:
        alpha: optional constant value for alpha.
        num_generator_per_critic: number of times to sample over the latent
            space before retraining the source critic. Default 4.
        budget: sampling budget.
        batch_size: batch size.
        z_bound: L-infinity sampling bound in the VAE latent space. Default 10.
        root: root directory containing the MNIST dataset.
        vae_ckpt: path to the trained MNIST VAE model checkpoint file.
        energy_ckpt: path to the trained surrogate image energy estimator file.
        device: device. Default CPU.
        savepath: optional path to save the optimization results to.
        seed: random seed. Default 42.
    Returns:
        None.
    """
    # Load the trained autoencoder.
    vae = VAE().to(device)
    vae.load_state_dict(torch.load(vae_ckpt))
    vae.eval()
    z_dim = vae.hidden_dims[-1]

    # Load the surrogate objective function.
    surrogate = Surrogate(z_dim).to(device)
    surrogate.load_state_dict(torch.load(energy_ckpt))
    surrogate.eval()

    # Define the sampling bounds.
    bounds = torch.tensor(
        [[-z_bound] * z_dim, [z_bound] * z_dim], device=device
    )

    # Initialize the sampling policy.
    ref_dataset = thv.datasets.MNIST(
        root,
        train=False,
        download=True,
        transform=thv.transforms.ToTensor()
    )
    policy = MNISTPolicy(
        bounds,
        ref_dataset,
        surrogate,
        alpha=alpha,
        batch_size=batch_size,
        z_dim=z_dim,
        device=device
    )

    # Choose the initial set of observations.
    a = []
    z = unnormalize(
        torch.rand(batch_size, z_dim, device=device), bounds=bounds
    )
    z_ref, _, _ = vae.encode(
        policy.reference_sample(8 * batch_size).flatten(start_dim=1)
    )
    policy.fit_critic(z.detach(), z_ref.detach())
    alpha = policy.alpha(z_ref)
    a.append(alpha)
    y = (1.0 - alpha) * surrogate(z) - alpha * torch.unsqueeze(
        policy.wasserstein(z_ref.detach(), z.detach()), dim=-1
    )
    y_gt = torch.unsqueeze(
        torch.mean(torch.square(vae.decode(z)), dim=-1), dim=-1
    )

    # Generative adversarial Bayesian optimization.
    for step in range(math.ceil(budget / batch_size) - 1):
        policy.fit(z.detach(), y.detach())

        # Optimize and get new observations.
        new_z = policy(y)

        # Train the source critic.
        z_ref, _, _ = vae.encode(
            policy.reference_sample(8 * batch_size).flatten(start_dim=1)
        )
        if step % num_generator_per_critic == 0:
            policy.fit_critic(z.detach(), z_ref.detach())

        # Calculate the surrogate and oracle objective values.
        alpha = policy.alpha(z_ref)
        a.append(alpha)
        new_y = (1.0 - alpha) * surrogate(new_z) - alpha * torch.unsqueeze(
            policy.wasserstein(z_ref.detach(), new_z.detach()), dim=-1
        )
        new_gt = torch.unsqueeze(
            torch.mean(torch.square(vae.decode(new_z)), dim=-1), dim=-1
        )

        # Update training points.
        z = torch.cat((z, new_z), dim=0)
        y = torch.cat((y, new_y), dim=0)
        y_gt = torch.cat((y_gt, new_gt), dim=0)

        # Update progress.
        policy.save_current_state_dict()

    # Save optimization results.
    if savepath is not None:
        X = vae.decode(z)
        X = X.reshape(-1, 1, *((int(math.sqrt(X.size(dim=-1))),) * 2))
        with open(savepath, mode="wb") as f:
            results = {
                "X": X.detach().cpu().numpy(),
                "z": z.detach().cpu().numpy(),
                "y": y.detach().cpu().numpy(),
                "y_gt": y_gt.detach().cpu().numpy(),
                "alpha": np.array(a),
                "batch_size": batch_size,
                "budget": budget
            }
            pickle.dump(results, f)
    idx = torch.argmax(y)
    print(y[idx], y_gt[idx])


if __name__ == "__main__":
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")

    parser = argparse.ArgumentParser(description="MNIST GABO Experiments")
    parser.add_argument(
        "--alpha", type=float, default=None, help="Optional constant alpha."
    )
    parser.add_argument(
        "--savepath", type=str, default=None, help="Path to save results to."
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed. Default 42."
    )
    args = parser.parse_args()

    torch.set_default_dtype(torch.float64)
    warnings.filterwarnings("ignore", category=BadInitialCandidatesWarning)
    seed_everything(args.seed)
    main(
        alpha=args.alpha,
        device=device,
        savepath=args.savepath,
        seed=args.seed,
        vae_ckpt=f"./mnist/checkpoints/mnist_vae_{args.seed}.pt",
        energy_ckpt=f"./mnist/checkpoints/mnist_surrogate_{args.seed}.pt"
    )
