"""
Evaluates a learned surrogate function for an objective in the molecule domain.

Author(s):
    Michael Yao @michael-s-yao

Licensed under the MIT License. Copyright University of Pennsylvania 2023.
"""
import json
import os
import matplotlib.pyplot as plt
import numpy as np
import pickle
import sys
import selfies as sf
import torch
from pathlib import Path
from tqdm import tqdm
from typing import Optional, Union

sys.path.append(".")
from selfies_vae.data import SELFIESDataModule
from selfies_vae.policy import BOPolicy
from selfies_vae.utils import MoleculeObjective
from models.fcnn import FCNN
from experiment.utility import plot_config


def eval_surrogate(
    surrogate_path: Union[Path, str],
    objective: str = "logP",
    savepath: Optional[Union[Path, str]] = None
) -> float:
    """
    Evaluates a learned surrogate function for an objective in the molecule
    domain.
    Input:
        surrogate_path: file path to the surrogate checkpoint file.
        objective: objective to evaluate against. Default `logP`.
        savepath: pickle path to save the evaluation results to. Default not
            saved.
    Returns:
        RMSE error between the surrogate and ground-truth functions over the
            test set.
    """
    dm = SELFIESDataModule()
    device = torch.device("cpu")

    vae = "./selfies_vae/ckpts/SELFIES-VAE-state-dict.pt"
    policy = BOPolicy(vae, device=device)

    objective = MoleculeObjective("logP")
    with open(
        os.path.join(os.path.dirname(__file__), "hparams.json"), "rb"
    ) as f:
        surrogate_hparams = json.load(f)
    surrogate = FCNN(
        in_dim=policy.vae.encoder_embedding_dim,
        out_dim=1,
        hidden_dims=surrogate_hparams["hidden_dims"],
        dropout=surrogate_hparams["dropout"],
        final_activation=None,
        hidden_activation="GELU"
    )
    surrogate.load_state_dict(
        torch.load(
            os.path.join(os.path.dirname(__file__), "./ckpts/27_surrogate.pt")
        )
    )
    surrogate = surrogate.to(device=device, dtype=policy.vae.dtype)

    val = {"preds": [], "gts": []}
    test = {"preds": [], "gts": []}
    for mol in tqdm(dm.val):
        smiles = sf.decoder(dm.val.decode(mol))
        val["gts"].append(objective(smiles))
        z = policy.encode([smiles]).detach().to(policy.vae.dtype)
        val["preds"].append(surrogate(z).item())
    for mol in tqdm(dm.test):
        smiles = sf.decoder(dm.test.decode(mol))
        test["gts"].append(objective(smiles))
        z = policy.encode([smiles]).detach().to(policy.vae.dtype)
        test["preds"].append(surrogate(z).item())
    if savepath is not None:
        with open(savepath, "wb") as f:
            pickle.dump({"val": val, "test": test}, f)

    error = np.array(test["preds"]) - np.array(test["gts"])
    return np.sqrt(np.mean(np.square(error)))


def plot_histogram(
    results_path: Union[Path, str],
    savepath: Optional[Union[Path, str]] = None,
    **kwargs
) -> None:
    """
    Plots the surrogate prediction results generated by the `eval_surrogate()`
    function.
    Input:
        results_path: file path to the pickle file with the results generated
            by the `eval_surrogate()` function.
        savepath: optional savepath to save the histogram plot to.
    Returns:
        None.
    """
    plot_config(**kwargs)
    with open(results_path, "rb") as f:
        results = pickle.load(f)
    val_results = np.array(results["val"]["preds"]) - np.array(
        results["val"]["gts"]
    )
    test_results = np.array(results["test"]["preds"]) - np.array(
        results["test"]["gts"]
    )
    plt.figure(figsize=(10, 4))
    bins = np.linspace(-10, 10, 50)
    plt.hist(
        val_results,
        bins=bins,
        density=True,
        label="Validation",
        alpha=0.7,
        color="#1A2A3C"
    )
    plt.hist(
        test_results,
        bins=bins,
        density=True,
        label="Test",
        alpha=0.5,
        color="#69AFDC"
    )
    plt.xlabel("Predicted Penalized LogP - True Penalized LogP")
    plt.ylabel("Probability Density")
    plt.legend()
    if savepath is None:
        plt.show()
    else:
        plt.savefig(savepath, dpi=600, transparent=True, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    surrogate_path = "./selfies_vae/ckpts/13_surrogate.pt"
    savepath = "./selfies_vae/docs/surrogate_results.pkl"
    eval_surrogate(surrogate_path=surrogate_path, savepath=savepath)
    plot_histogram(
        savepath,
        savepath="./selfies_vae/docs/surrogate_results.png",
        fontsize=20
    )
