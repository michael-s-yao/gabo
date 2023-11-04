"""
Builds a penalized logP objective estimator.

Author(s):
    Michael Yao @michael-s-yao

Licensed under the MIT License. Copyright University of Pennsylvania 2023.
"""
import json
import pickle
import os
import sys
import selfies as sf
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm
from typing import Optional, Union

sys.path.append(".")
from models.fcnn import FCNN
from selfies_vae.data import SELFIESDataModule
from selfies_vae.vae import InfoTransformerVAE
from selfies_vae.utils import MoleculeObjective


def build_objective(
    hparams: Union[Path, str] = "./selfies_vae/hparams.json",
    cache_fn: Union[Path, str] = "./selfies_vae/cache.pkl",
    seed: int = 42,
    plotpath: Optional[Union[Path, str]] = None,
    savepath: Optional[Union[Path, str]] = None,
    device: torch.device = torch.device("cuda")
) -> float:
    """
    Trains and tests an MLP regressor model as a logP objective estimator.
    Input:
        hparams: the file path to the JSON file with the model hyperparameters.
        seed: random seed. Default 42.
        plotpath: optional path to save the histogram plot to. Default None.
        savepath: optional path to save the model to. Default None.
    Returns:
        RMSE value on the test dataset.
    """
    with open(hparams, "rb") as f:
        hparams = json.load(f)
    dm = SELFIESDataModule(
        batch_size=hparams["batch_size"], load_train_data=True, num_workers=0
    )
    vae = InfoTransformerVAE(dm.train).to(device)
    vae.load_state_dict(
        torch.load(
            "./selfies_vae/ckpts/SELFIES-VAE-state-dict.pt",
            map_location=device
        ),
        strict=True
    )
    vae.eval()

    objective = MoleculeObjective("logP")
    surrogate = FCNN(
        in_dim=vae.encoder_embedding_dim,
        out_dim=1,
        hidden_dims=hparams["hidden_dims"],
        dropout=hparams["dropout"],
        final_activation=None,
        hidden_activation="ReLU"
    )
    if hparams["optimizer"].title() == "Adam":
        optimizer = optim.Adam(
            surrogate.parameters(),
            hparams["lr"],
            weight_decay=hparams["weight_decay"]
        )
    elif hparams["optimizer"].upper() == "SGD":
        optimizer = optim.SGD(
            surrogate.params(),
            hparams["lr"],
            weight_decay=hparams["weight_decay"],
            momentum=0.9,
            nesterov=True
        )
    mse_loss, val_loss = nn.MSELoss(), None
    best_loss, best_epoch = 1e12, -1
    cache, loaded_cache = {}, False
    if os.path.isfile(cache_fn):
        with open(cache_fn, "rb") as f:
            cache = pickle.load(f)
        loaded_cache = True

    for epoch in range(hparams["num_epochs"]):
        surrogate.train()
        with tqdm(
            dm.train_dataloader(), desc=f"Epoch {epoch}", leave=False
        ) as pbar:
            for mol in pbar:
                z = vae(mol.to(device))["z"].reshape(
                    -1, vae.encoder_embedding_dim
                )
                z = torch.squeeze(z, dim=0).detach()
                smiles = [
                    sf.decoder(m)
                    for m in [dm.train.decode(tok) for tok in mol]
                ]

                if not loaded_cache and epoch == 0:
                    y = [objective(smi) for smi in smiles]
                    for smi, val in zip(smiles, y):
                        cache[smi] = val
                else:
                    y = [cache[smi] for smi in smiles]

                y = torch.tensor(y).to(z)
                loss = mse_loss(torch.squeeze(surrogate(z), dim=-1), y)
                loss.backward()
                pbar.set_postfix(train_loss=loss.item(), val_loss=val_loss)
                optimizer.step()

        surrogate.eval()
        val_loss = []
        for mol in tqdm(dm.val_dataloader(), desc="Validating", leave=False):
            z = vae(mol)["z"].reshape(-1, vae.encoder_embedding_dim)
            z = torch.squeeze(z, dim=0).detach()
            smiles = [
                sf.decoder(m) for m in [dm.train.decode(tok) for tok in mol]
            ]
            if not loaded_cache and epoch == 0:
                y = [objective(smi) for smi in smiles]
                for smi, val in zip(smiles, y):
                    cache[smi] = val
            else:
                y = [cache[smi] for smi in smiles]
            y = torch.tensor(y).to(z)
            val_loss.append(
                mse_loss(torch.squeeze(surrogate(z), dim=-1), y).item()
            )
        val_loss = sum(val_loss) / len(val_loss)
        if val_loss < best_loss:
            best_loss, best_epoch = val_loss, epoch
            torch.save(surrogate.state_dict(), f"./ckpts/{epoch}_surrogate.pt")
        if epoch - best_epoch > hparams["patience"] > 0:
            break

        if not loaded_cache:
            with open(cache_fn, "wb") as f:
                pickle.dump(cache, f)


if __name__ == "__main__":
    build_objective()
