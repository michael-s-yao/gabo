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
from digits.surrogate import SurrogateObjective
from models.dual import Alpha
from models.transform import SobelGaussianTransform
from digits.policy import MNISTPolicy
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
        type=float,
        default=None,
        help="An optional constant value for alpha between 0 and 1."
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
        default=2048,
        help="Sampling budget. Default 2048. Use -1 for infinite budget."
    )
    parser.add_argument(
        "--batch_size", type=int, default=64, help="Batch size. Default 64."
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

    dm = MNISTDataModule(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed
    )
    dm.prepare_data()
    dm.setup()

    surrogate = SurrogateObjective.load_from_checkpoint(args.surrogate)
    surrogate = surrogate.to(device)
    surrogate.eval()

    alpha = Alpha(surrogate, constant=args.alpha)
    policy = MNISTPolicy(
        dm.test, surrogate, device=device, seed=args.seed
    )

    z_ref = policy.reference_sample(args.batch_size)
    z_ref = z_ref.detach()
    z_mean, z_std = torch.mean(z_ref).detach(), torch.std(z_ref).detach()
    ztransform = SobelGaussianTransform(z_mean, z_std)
    sobol = SobolEngine(
        dimension=np.prod(surrogate.convae.model.z_dim),
        scramble=True,
        seed=args.seed
    )
    zsobel = sobol.draw(n=args.batch_size).to(device).detach()
    zgauss = ztransform(zsobel)
    X = policy.decode(ztransform(zsobel))

    history = []
    alpha.fit_critic(z_ref, zgauss)
    y = surrogate(zgauss).detach()
    y_gt = policy.oracle(zgauss)
    history.append(alpha(z_ref))

    likelihood = GaussianLikelihood().to(device)
    covar_module = ScaleKernel(MaternKernel(nu=2.5)).to(device)
    a = history[-1]
    model = SingleTaskVariationalGP(
        zsobel,
        ((1 - a) * y.detach()) + a * torch.unsqueeze(
            alpha.penalize(z_ref, zgauss).detach(), dim=-1
        ),
        likelihood=likelihood,
        covar_module=covar_module
    )
    mll = PredictiveLogLikelihood(
        likelihood, model.model, num_data=zgauss.size(dim=0)
    )

    # Generative adversarial Bayesian optimization.
    budget = np.inf if args.budget < 1 else args.budget
    while len(y) < budget and not policy.restart_triggered:
        fit_gpytorch_mll(mll)
        z_ref = policy.reference_sample(args.batch_size)
        z_ref = z_ref.detach()
        z_next = policy(
            model,
            zsobel,
            y,
            batch_size=args.batch_size
        )
        z_next = z_next.detach()
        z_next_gaussian = ztransform(z_next)
        X_next = policy.decode(z_next_gaussian)
        X = torch.cat((X, X_next), dim=0)
        zsobel = torch.cat((zsobel, z_next), dim=0)

        alpha.fit_critic(z_ref, z_next_gaussian)
        history.append(alpha(z_ref))

        y_next = surrogate(z_next_gaussian)
        y_gt_next = policy.oracle(z_next_gaussian)

        a = history[-1]
        policy.update_state(
            ((1 - a) * y_next) + a * torch.unsqueeze(
                alpha.penalize(z_ref, z_next_gaussian), dim=-1
            )
        )
        y = torch.cat((y, y_next), dim=0)
        y_gt = torch.cat((y_gt, y_gt_next), dim=0)
        best_idx = torch.argmax(torch.squeeze(y, dim=-1))
        print(
            f"{len(zsobel)}) Best value: {y[best_idx].item():.5f} |",
            f"(Oracle: {y_gt[best_idx].item():.5f}) |",
            f"(alpha: {history[-1]})"
        )

    if args.savepath is None:
        return
    with open(args.savepath, "wb") as f:
        results = {
            "batch_size": args.batch_size,
            "alpha": history,
            "X": X.detach().cpu().numpy(),
            "z": ztransform(zsobel.detach()).cpu().numpy(),
            "y": y.detach().cpu().numpy(),
            "y_gt": y_gt.detach().cpu().numpy()
        }
        pickle.dump(results, f)


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    main()
