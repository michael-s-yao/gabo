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
import torch
from pathlib import Path
from typing import Optional, Union

sys.path.append(".")
from digits.mnist import MNISTDataModule
from experiment.utility import seed_everything


def main(seed: int = 42, savepath: Optional[Union[Path, str]] = None):
    seed_everything(seed=seed)
    dm = MNISTDataModule(seed=seed, num_workers=0)
    dm.prepare_data()
    dm.setup()

    train_y, test_y = [], []
    for X, _ in dm.train:
        train_y.append(torch.mean(torch.square(X)).item())
    for X, _ in dm.test:
        test_y.append(torch.mean(torch.square(X)).item())

    plt.plot(figsize=(10, 5))
    bins = np.linspace(0, 1, 100)
    plt.hist(
        train_y, bins=bins, density=True, alpha=0.7, label="Training Dataset"
    )
    plt.hist(test_y, bins=bins, density=True, alpha=0.5, label="Test Dataset")
    plt.xlabel(r"Energy $||x||_2^2$")
    plt.ylabel("Probability Density")
    percentile_95 = 0.5 * (
        sorted(test_y)[round(0.95 * len(test_y))] + sorted(train_y)[
            round(0.95 * len(train_y))
        ]
    )
    percentile_100 = 0.5 * (sorted(test_y)[-1] + sorted(train_y)[-1])
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


if __name__ == "__main__":
    main(savepath="./digits/docs/distribution.png")
