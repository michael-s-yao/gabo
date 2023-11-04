"""
Bayesian optimization over molecular latent space for generative adversarial
optimization.

Author(s):
    Yimeng Zeng @yimeng-zeng
    Michael Yao @michael-s-yao

Licensed under the MIT License. Copyright University of Pennsylvania 2023.
"""
import argparse
import math
import sys
import selfies as sf
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
from selfies_vae.data import SELFIESDataModule
from selfies_vae.policy import BOPolicy
from selfies_vae.utils import MoleculeObjective, smiles_to_tokens
from models.objective import SELFIESObjective
from experiment.utility import seed_everything, get_device


def build_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Molecule Objective Optimization"
    )

    parser.add_argument(
        "--model",
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
        "--batch_size", type=int, default=128, help="Batch size. Default 128."
    )
    parser.add_argument(
        "--device", type=str, default="auto", help="Device. Default `auto`."
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed. Default 42."
    )

    return parser.parse_args()


def eval_surrogate():
    dm = SELFIESDataModule()
    objective = MoleculeObjective("logP")
    surrogate = SELFIESObjective(
        dm.val.vocab2idx, "./MolOOD/checkpoints/regressor.ckpt"
    )
    squared_error = []
    for mol in dm.val:
        gt = objective(sf.decoder(dm.val.decode(mol)))
        pred = surrogate(torch.unsqueeze(mol, dim=0)).item()
        squared_error.append((gt - pred) * (gt - pred))
    return math.sqrt(sum(squared_error) / len(squared_error))


def main():
    args = build_args()
    seed_everything(seed=args.seed)
    device = get_device(args.device)
    warnings.filterwarnings("ignore", category=BadInitialCandidatesWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    policy = BOPolicy(args.model, device=device)
    objective = MoleculeObjective(args.objective)
    surrogate = SELFIESObjective(
        policy.dataset.vocab2idx, "./MolOOD/checkpoints/regressor.ckpt"
    )

    sobol = SobolEngine(
        dimension=policy.vae.encoder_embedding_dim,
        scramble=True,
        seed=args.seed
    )
    z_init = sobol.draw(n=args.batch_size).to(
        dtype=torch.float32, device=policy.vae.device
    )

    smiles = policy.decode(z_init)
    z = policy.encode(smiles).detach().to(torch.double)
    y = torch.tensor(
        [[objective(smi)] for smi in smiles],
        dtype=torch.double,
        device=policy.vae.device
    )
    y_mean, y_std = torch.mean(y), torch.std(y)

    likelihood = GaussianLikelihood().to(policy.vae.device)
    covar_module = ScaleKernel(MaternKernel(nu=2.5)).to(policy.vae.device)
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

    while not policy.restart_triggered:
        fit_gpytorch_mll(mll)
        z_next = policy(model, z, y, batch_size=10)

        # Decode batch to smiles, get logP values.
        smiles = policy.decode(z_next)
        y_next = [
            surrogate(smiles_to_tokens(smiles, policy.dataset.vocab2idx))
        ]
        y_next = [objective(smi) for smi in smiles]
        samples = [
            (zz, yy) for zz, yy in zip(z_next, y_next) if yy is not None
        ]
        z_next = torch.cat([torch.unsqueeze(zz, dim=0) for zz, _ in samples])
        y_next = torch.cat([torch.tensor([yy]) for _, yy in samples]).to(
            dtype=policy.vae.dtype, device=policy.vae.device
        )
        y_next = torch.unsqueeze(y_next, dim=-1)

        policy.update_state(y=y_next)
        z, y = torch.cat((z, z_next), dim=-2), torch.cat((y, y_next), dim=-2)
        print(
            f"{len(z)}) Best value: {policy.state.best_value:.5f}"
        )


if __name__ == "__main__":
    print(eval_surrogate())
    # main()
