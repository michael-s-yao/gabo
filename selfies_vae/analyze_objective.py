"""
Script to explore the distribution of objective values in the training and
test distributions.

Author(s):
    Michael Yao @michael-s-yao

Licensed under the MIT License. Copyright University of Pennsylvania 2023.
"""
import sys
import matplotlib.pyplot as plt
import numpy as np
import pickle
import selfies as sf
from pathlib import Path
from tqdm import tqdm
from typing import Optional, Union

sys.path.append(".")
from selfies_vae.data import SELFIESDataModule
from selfies_vae.utils import MoleculeObjective
from experiment.utility import seed_everything


def plot_results(
    results: Union[Path, str], savepath: Optional[Union[Path, str]] = None
) -> None:
    with open(results, "rb") as f:
        logP = pickle.load(f)
    train, test = sorted(logP["train"]), sorted(logP["test"])

    plt.plot(figsize=(10, 5))
    bins = np.linspace(-20, 20, 100)
    plt.hist(
        logP["train"],
        bins=bins,
        density=True,
        alpha=0.7,
        label="Training Dataset"
    )
    plt.hist(
        logP["test"],
        bins=bins,
        density=True,
        alpha=0.5,
        label="Test Dataset"
    )
    plt.xlabel("Penalized logP Value")
    plt.ylabel("Probability Density")
    idx95_train, idx95_test = round(0.95 * len(train)), round(0.95 * len(test))
    percentile_95 = 0.5 * (train[idx95_train] + test[idx95_test])
    percentile_100 = max(train[-1], test[-1])
    plt.axvline(
        percentile_95,
        color="k",
        alpha=0.5,
        ls="--",
        label=f"95th Percentile: {percentile_95:.3f}"
    )
    plt.axvline(
        percentile_100,
        color="k",
        alpha=1.0,
        ls="--",
        label=f"100th Percentile: {percentile_100:.3f}"
    )
    plt.legend()
    if savepath is None:
        plt.show()
    else:
        plt.savefig(savepath, dpi=600, transparent=True, bbox_inches="tight")
    plt.close()


def main(savepath: Union[Path, str], seed: int = 42):
    seed_everything(seed=seed)
    dm = SELFIESDataModule(num_workers=0, load_train_data=True)
    dm.prepare_data()
    dm.setup(None)

    objective = MoleculeObjective("logP")

    train_logP, test_logP = [], []
    for mol in tqdm(dm.train, desc="Train Dataset", leave=False):
        train_logP.append(objective(sf.decoder(dm.train.decode(mol))))
    for mol in tqdm(dm.test, desc="Test Dataset", leave=False):
        test_logP.append(objective(sf.decoder(dm.test.decode(mol))))
    with open(savepath, "wb") as f:
        pickle.dump({"train": train_logP, "test": test_logP}, f)


if __name__ == "__main__":
    results = "./selfies_vae/docs/logP_distribution.pkl"
    # main(savepath=results)
    plot_results(
        results=results, savepath="./selfies_vae/docs/logP_distribution.png"
    )
