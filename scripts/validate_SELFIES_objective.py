"""
Script to interrogate the accuracy of the learned objective function that
estimates the log P value of an input molecule.

Author(s):
    Yimeng Zeng @yimeng-zeng
    Michael Yao @michael-s-yao

Licensed under the MIT License. Copyright University of Pennsylvania 2023.
"""
import os
import numpy as np
import sys
import pandas as pd
import torch
from pathlib import Path
from tqdm import tqdm
from typing import Union

sys.path.append(".")
sys.path.append("MolOOD")
from models.objective import SELFIESObjective
from mol import load_vocab
from MolOOD.molformers.datamodules.logp_dataset import LogPDataModule
from experiment.utility import get_device


def validate_SELFIES_objective(
    datadir: Union[Path, str] = "./MolOOD/data",
    surrogate_ckpt: Union[Path, str] = "./MolOOD/checkpoints/regressor.ckpt",
    batch_size: int = 1_024,
    num_workers: int = 0,
    device: str = "auto"
) -> float:
    """
    Validate the implementation of the SELFIES objective function.
    Input:
        datadir: data directory. Default `./MolOOD/data`.
        surrogate_ckpt: path to model ckpt for objective.
            Default `./MolOOD/checkpoints/regressor.ckpt`.
        batch_size: batch size. Default 1,024.
        num_workers: number of workers. Default 0.
        device: device to run validation on. Default `auto`.
    Returns:
        RMSE between the estimated log P value and true log P value.
    """
    device = get_device(device)
    vocab = load_vocab(os.path.join(datadir, "vocab.json"))
    datamodule = LogPDataModule(
        datadir, vocab=vocab, batch_size=batch_size, num_workers=num_workers
    )

    objective = SELFIESObjective(vocab=vocab, surrogate_ckpt=surrogate_ckpt)
    objective = objective.to(device)

    target = pd.read_csv(
        os.path.join(datadir, "logp_scores", "test_logp.csv"), header=None
    )
    target = np.squeeze(target.values, axis=-1)

    all_preds = np.array([])
    with torch.inference_mode():
        for batch in tqdm(
            datamodule.test_dataloader(),
            desc="SELFIES log P Objective Validation",
            leave=False
        ):
            tokens, _ = batch
            tokens = tokens.to(device)
            all_preds = np.concatenate((
                all_preds, objective(tokens).detach().cpu().numpy()
            ))
    return np.sqrt(np.mean(np.square(all_preds - target)))


if __name__ == "__main__":
    print(f"RMSE: {validate_SELFIES_objective()}")
