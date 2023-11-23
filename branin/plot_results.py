"""
Results analyzer script for toy Branin task experiments.

Author(s):
    Michael Yao

Licensed under the MIT License. Copyright University of Pennsylvania 2023.
"""
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
from pathlib import Path
from typing import Optional, Union


def main(
    results_dir: Union[Path, str] = "branin/docs",
    savepath: Optional[Union[Path, str]] = None
):
    directory, _, fns = next(os.walk(results_dir))
    fns = [os.path.join(directory, f) for f in fns]
    alphas = {a: [] for a in ["0.0", "0.2", "0.5", "0.8", "1.0", "Ours"]}
    for f in fns:
        key = f.split("=")[1].split("_")[0]
        if "pkl" in key:
            alphas["Ours"].append(f)
        elif key in alphas.keys():
            alphas[key].append(f)

    print("     |    Surrogate   |     Oracle")
    print("----------------------------------------")
    for a, fns in alphas.items():
        surr, best = [], []
        for pkl in fns:
            with open(pkl, "rb") as f:
                results = pickle.load(f)
            y, y_gt = results["y"], results["y_gt"]
            batch_size = results["batch_size"]
            idxs = np.argsort(y)
            surr += y[idxs][-(batch_size // 4):].tolist()
            best += y_gt[idxs][-(batch_size // 4):].tolist()
        best = np.array(best)

        padding = " " if np.mean(surr) > 0.0 else ""
        a = a + " " if a != "Ours" else a
        print(
            f"{a} | {padding}{np.mean(surr):.2f} +/- {np.std(surr):.2f} |",
            f"{np.mean(best):.2f} +/- {np.std(best):.2f}"
        )

    a = []
    for pkl in alphas["Ours"]:
        with open(pkl, "rb") as f:
            results = pickle.load(f)
        a += results["alpha"].tolist()
    a = np.array(a).reshape(len(alphas["Ours"]), -1)
    plt.figure(figsize=(10, 5))
    plt.plot(1 + np.arange(a.shape[-1]), np.mean(a, axis=0), color="k")
    plt.xlabel("Optimization Step")
    plt.ylabel(r"$\alpha$")
    plt.ylim(0.0, 1.0)
    if savepath is None:
        plt.show()
    else:
        plt.savefig(savepath, dpi=600, bbox_inches="tight", transparent=True)
    plt.close()


if __name__ == "__main__":
    main()
