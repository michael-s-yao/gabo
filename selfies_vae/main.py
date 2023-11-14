"""
Bayesian optimization over molecular latent space for generative adversarial
optimization.

Author(s):
    Yimeng Zeng @yimeng-zeng
    Michael Yao @michael-s-yao

Licensed under the MIT License. Copyright University of Pennsylvania 2023.
"""
import argparse
import json
import numpy as np
import os
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
from botorch.exceptions.warnings import BadInitialCandidatesWarning

sys.path.append(".")
from selfies_vae.vae import InfoTransformerVAE
from selfies_vae.data import SELFIESDataset
from selfies_vae.policy import SELFIESAdversarialPolicy
from selfies_vae.utils import MoleculeObjective
from models.fcnn import FCNN
from experiment.utility import seed_everything


def build_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Molecule Objective Optimization"
    )

    parser.add_argument(
        "--alpha",
        type=str,
        required=True,
        help="A float between 0 and 1, or `Lipschitz` for our method."
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=0,
        help="Number of critic warmup steps before starting optimization."
    )
    parser.add_argument(
        "--encoder",
        type=str,
        default="./selfies_vae/ckpts/SELFIES-VAE-state-dict.pt",
        help="Path to trained transformer VAE model."
    )
    parser.add_argument(
        "--objective",
        type=str,
        default="logP",
        choices=list(MoleculeObjective().guacamol_objs.keys()),
        help="Objective function to maximize. Default penalized logP score."
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

    return parser.parse_args()


def main():
    args = build_args()
    seed_everything(seed=args.seed, use_deterministic=False)
    device = torch.device("cpu")
    warnings.filterwarnings("ignore", category=BadInitialCandidatesWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=ResourceWarning)

    dataset = SELFIESDataset(
        os.path.join("./selfies_vae/data/test_selfie.gz"), load_data=True
    )

    vae = InfoTransformerVAE(SELFIESDataset()).to(device)
    vae.load_state_dict(
        torch.load(args.encoder, map_location=device), strict=True
    )
    vae.eval()
    objective = MoleculeObjective(args.objective)
    with open(
        os.path.join(os.path.dirname(__file__), "hparams.json"), "rb"
    ) as f:
        surrogate_hparams = json.load(f)
    surrogate = FCNN(
        in_dim=vae.encoder_embedding_dim,
        out_dim=1,
        hidden_dims=surrogate_hparams["hidden_dims"],
        dropout=surrogate_hparams["dropout"],
        final_activation=None,
        hidden_activation="ReLU"
    )
    surrogate.load_state_dict(
        torch.load(
            os.path.join(os.path.dirname(__file__), "ckpts", "13_surrogate.pt")
        )
    )
    surrogate = surrogate.to(device=device, dtype=vae.dtype)
    policy = SELFIESAdversarialPolicy(
        ref_dataset=dataset,
        vae=vae,
        alpha=args.alpha,
        surrogate=surrogate,
        device=device
    )

    z_ref = policy.reference_sample(8 * args.batch_size)
    z_mean, z_std = torch.mean(z_ref), torch.std(z_ref)
    sobol = SobolEngine(
        dimension=policy.z_dim,
        scramble=True,
        seed=args.seed
    )
    z_init = z_mean + (z_std * sobol.draw(n=(8 * args.batch_size)).to(device))

    corr_factors = []
    smiles = policy.decode(z_init)
    z = policy.encode(smiles).detach().to(vae.dtype)
    y = surrogate(z)
    y, alpha = policy.penalize(y, z)
    y = y.detach()
    if not isinstance(alpha, float):
        alpha = alpha.detach().cpu().numpy()
    corr_factors.append(alpha)

    y_gt = torch.tensor([[objective(smi)] for smi in smiles])
    y_mean, y_std = torch.mean(y), torch.std(y)

    likelihood = GaussianLikelihood().to(vae.device)
    covar_module = ScaleKernel(MaternKernel(nu=2.5)).to(vae.device)
    model = SingleTaskVariationalGP(
        z,
        (y - y_mean) / y_std,
        inducing_points=1024,
        likelihood=likelihood,
        covar_module=covar_module
    )
    mll = PredictiveLogLikelihood(
        likelihood, model.model, num_data=z.size(dim=0)
    )

    # Warmup source critic training.
    for _ in range(args.warmup_steps):
        policy.update_critic(model, z, y)
    # Generative adversarial Bayesian optimization.
    budget = np.inf if args.budget < 1 else args.budget
    while len(y) < budget and not policy.restart_triggered:
        fit_gpytorch_mll(mll)
        z_next = policy(model, z, y, batch_size=args.batch_size)

        smiles = policy.decode(z_next)
        y_next = torch.squeeze(surrogate(z_next), dim=-1)
        y_next_gt = [objective(smi) for smi in smiles]
        samples = [
            (zz, ypred, ygt)
            for zz, ypred, ygt in zip(z_next, y_next, y_next_gt)
            if ypred is not None and ygt is not None
        ]
        z_next = torch.cat([
            torch.unsqueeze(zz, dim=0) for zz, _, _ in samples
        ])
        y_next = torch.cat([torch.tensor([yy]) for _, yy, _ in samples]).to(
            dtype=vae.dtype, device=device
        )
        y_next_gt = torch.cat([torch.tensor([yy]) for _, _, yy in samples]).to(
            dtype=vae.dtype, device=device
        )
        y_next = torch.unsqueeze(y_next, dim=-1)
        y_next_gt = torch.unsqueeze(y_next_gt, dim=-1)

        y_next, alpha = policy.penalize(y_next, z_next)
        if not isinstance(alpha, float):
            alpha = alpha.detach().cpu().numpy()
        corr_factors.append(alpha)

        policy.update_state(y_next)
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
            "alpha": corr_factors,
            "z": z.detach().cpu().numpy(),
            "y": y.detach().cpu().numpy(),
            "y_gt": y_gt.detach().cpu().numpy()
        }
        pickle.dump(results, f)


if __name__ == "__main__":
    main()
