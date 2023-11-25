"""
Analyze results from molecule generative optimization.

Author(s):
    Michael Yao

Licensed under the MIT License. Copyright University of Pennsylvania 2023.
"""
import numpy as np
import pickle
import sys
import selfies as sf
import torch
import warnings
from pathlib import Path
from typing import Union

sys.path.append(".")
from molecules.data import SELFIESDataModule
from molecules.vae import InfoTransformerVAE
from molecules.utils import MoleculeObjective
from fcd_torch import FCD


def main(
    results_path: Union[Path, str],
    vae_ckpt: Union[Path, str] = "./molecules/checkpoints/vae.pt"
):
    with open(results_path, "rb") as f:
        results = pickle.load(f)
    y, y_gt = np.squeeze(results["y"]), np.squeeze(results["y_gt"])
    batch_size = results["batch_size"]

    # Load the trained autoencoder.
    vae = InfoTransformerVAE()
    vae.load_state_dict(torch.load(vae_ckpt))
    vae.eval()

    print(results_path)
    idxs = np.argsort(y)
    print(f"  Best Surrogate: {y[idxs][-1]} (Oracle: {y_gt[idxs][-1]})")
    y_batch, y_gt_batch = y[idxs][-batch_size:], y_gt[idxs][-batch_size:]
    print(
        f"  Best {batch_size} Surrogates:",
        f"{np.mean(y_batch)} +/- {np.std(y_batch)}",
        f"(Oracle: {np.mean(y_gt_batch)} +/- {np.std(y_gt_batch)})"
    )

    fcd = FCD(device="cpu")
    dm = SELFIESDataModule(num_workers=0)
    dm.prepare_data()
    dm.setup(None)
    P = next(iter(dm.test_dataloader()))[torch.from_numpy(idxs)][-batch_size:]
    P = [sf.decoder(dm.test.decode(tok)) for tok in P]
    z = torch.from_numpy(results["z"].reshape(results["z"].shape[0], 2, -1))
    z = z[torch.from_numpy(idxs)][-batch_size:]
    Q = [sf.decoder(dm.test.decode(tok)) for tok in vae.sample(z=z)]
    Q = [q for q in Q if len(q)]
    P = P[:len(Q)]
    fcd(P, Q)


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    warnings.filterwarnings("ignore", category=UserWarning)
    # main("./molecules/docs/alpha=0.0.pkl")
    # main("./molecules/docs/alpha=0.2.pkl")
    # main("./molecules/docs/alpha=0.5.pkl")
    # main("./molecules/docs/alpha=0.8.pkl")
    # main("./molecules/docs/alpha=1.0.pkl")
    main("./molecules/docs/alpha.pkl")
