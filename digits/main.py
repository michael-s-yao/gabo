"""
Main driver program for MNIST-related training.

Author(s):
    Michael Yao

Licensed under the MIT License. Copyright University of Pennsylvania 2023.
"""
import argparse
import numpy as np
import pickle
import sys
import torch
import warnings
from torch.quasirandom import SobolEngine
from gpytorch.mlls import PredictiveLogLikelihood
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from botorch.fit import fit_gpytorch_mll
from botorch.models.approximate_gp import SingleTaskVariationalGP
from botorch.exceptions.warnings import (
    BadInitialCandidatesWarning, InputDataWarning
)

sys.path.append(".")
from digits.mnist import MNISTDataModule
from digits.policy import MNISTAdversarialPolicy
from digits.surrogate import SurrogateObjective
from models.convae import ConvAutoEncLightningModule
from experiment.utility import seed_everything


def build_args() -> argparse.Namespace:
    """
    Builds the relevant parameters for the MNIST digit image generation task.
    Input:
        None.
    Returns:
        A namespace with the relevant parameters for the MNIST experiments.
    """
    parser = argparse.ArgumentParser(
        description="MNIST Generative Adversarial Optimization Experiments"
    )

    parser.add_argument(
        "--alpha",
        type=str,
        required=True,
        help="A float between 0 and 1, or `Lipschitz` for our method."
    )
    parser.add_argument(
        "--autoencoder",
        type=str,
        default="./digits/ckpts/convae.ckpt",
        help="Path to trained convolutional autoencoder model."
    )
    parser.add_argument(
        "--surrogate",
        type=str,
        default="./digits/ckpts/surrogate.ckpt",
        help="Path to trained surrogate objective model."
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=1024,
        help="Sampling budget. Default 1024. Use -1 for infinite budget."
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Batch size. Default 16."
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed. Default 42."
    )
    parser.add_argument(
        "--savepath",
        type=str,
        default=None,
        help="Path to the save the model results to. Default not saved."
    )
    parser.add_argument(
        "--num_workers",
        default=0,
        type=int,
        help="Number of workers. Default 0."
    )

    return parser.parse_args()


def main():
    args = build_args()
    seed_everything(seed=args.seed, use_deterministic=False)
    device = torch.device("cuda:0")
    warnings.filterwarnings("ignore", category=BadInitialCandidatesWarning)
    warnings.filterwarnings("ignore", category=InputDataWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    dm = MNISTDataModule(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed
    )
    dm.prepare_data()
    dm.setup()

    convae = ConvAutoEncLightningModule.load_from_checkpoint(args.autoencoder)
    convae = convae.to(device)
    convae.eval()
    surrogate = SurrogateObjective.load_from_checkpoint(args.surrogate)
    surrogate = surrogate.to(device)
    surrogate.eval()

    policy = MNISTAdversarialPolicy(
        dm.test, convae, args.alpha, surrogate, device, seed=args.seed
    )

    z_ref = policy.encode(
        policy.reference_sample(8 * args.batch_size).reshape(-1, *policy.x_dim)
    )[0]
    z_mean, z_std = torch.mean(z_ref), torch.std(z_ref)
    sobol = SobolEngine(
        dimension=np.prod(policy.z_dim),
        scramble=True,
        seed=args.seed
    )
    z_init = z_mean + (z_std * sobol.draw(n=args.batch_size).to(device))

    X = policy.decode(z_init)
    z, _ = policy.encode(X)
    z = z.detach().to(convae.dtype)
    y = surrogate(X.flatten(start_dim=1))
    y = policy.penalize(y, X.flatten(start_dim=1)).detach()

    y_gt = torch.tensor([torch.mean(torch.square(x)) for x in X]).to(device)
    y_gt = torch.unsqueeze(y_gt, dim=-1)

    likelihood = GaussianLikelihood().to(device)
    covar_module = ScaleKernel(MaternKernel(nu=2.5)).to(device)
    model = SingleTaskVariationalGP(
        z,
        y,
        inducing_points=1024,
        likelihood=likelihood,
        covar_module=covar_module
    )
    mll = PredictiveLogLikelihood(
        likelihood, model.model, num_data=z.size(dim=0)
    )

    budget = np.inf if args.budget < 1 else args.budget
    while len(y) < budget and not policy.restart_triggered:
        fit_gpytorch_mll(mll)
        z_next = policy(model, z, y, batch_size=args.batch_size)

        X_next = policy.decode(z_next)
        y_next = torch.squeeze(surrogate(X_next.flatten(start_dim=1)), dim=-1)
        y_next_gt = torch.tensor([torch.mean(torch.square(x)) for x in X_next])
        y_next = torch.unsqueeze(y_next, dim=-1)
        y_next_gt = torch.unsqueeze(y_next_gt.to(device), dim=-1)

        y_next = policy.penalize(y_next, X_next.flatten(start_dim=1))
        policy.update_state(y_next)
        X = torch.cat((X, X_next), dim=0)
        z = torch.cat((z, z_next), dim=-2)
        y = torch.cat((y, y_next), dim=-2)
        y_gt = torch.cat((y_gt, y_next_gt), dim=-2)
        print(
            f"{len(z)}) Best value: {torch.max(y).item():.5f} |",
            f"(Oracle: {torch.max(y_gt).item():.5f})"
        )

        policy.update_critic(model, z, y)

    if args.savepath is None:
        return
    with open(args.savepath, "wb") as f:
        results = {
            "batch_size": args.batch_size,
            "X": X.detach().cpu().numpy(),
            "z": z.detach().cpu().numpy(),
            "y": y.detach().cpu().numpy(),
            "y_gt": y_gt.detach().cpu().numpy()
        }
        pickle.dump(results, f)


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    main()
