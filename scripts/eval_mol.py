"""
Molecule generation model evaluation script and helper functions.

Author(s):
    Michael Yao

Licensed under the MIT License. Copyright University of Pennsylvania 2023.
"""
import argparse
import matplotlib.pyplot as plt
import numpy as np
import Levenshtein
from rdkit import Chem, DataStructs
from rdkit.Chem.Fingerprints import FingerprintMols
import selfies as sf
import sys
from pathlib import Path
from tqdm import tqdm
import torch
import torch.nn.functional as F
from typing import Any, Dict, Optional, Sequence, Union

sys.path.append(".")
sys.path.append("MolOOD")
from data.molecule import (
    SELFIESDataModule, tokens_to_selfies, one_hot_encodings_to_selfies
)
from models.objective import SELFIESObjective
from models.vae import SELFIESVAEModule
from experiment.utility import seed_everything, plot_config, get_device
from mol import load_vocab
from MolOOD.mol_oracle.mol_utils import logp_wrapper


def build_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Molecule Generation Model Evaluation"
    )

    parser.add_argument(
        "--model", required=True, type=str, help="Generator model checkpoint."
    )
    parser.add_argument(
        "--num_molecules",
        default=256,
        type=int,
        help="Number of molecules to evaluate. Default 256."
    )
    parser.add_argument(
        "--disable_plot",
        action="store_true",
        help="If True, no plots are generated."
    )
    parser.add_argument(
        "--savepath",
        default=None,
        type=str,
        help="File path to save the plot. Default not saved."
    )
    parser.add_argument(
        "--metric",
        type=str,
        choices=(
            "tanimoto",
            "levenshtein",
            "diversity",
            "learned_objective",
            "oracle"
        ),
        default="tanimoto",
        help="Metric for molecule similarity. Default Tanimoto similarity."
    )
    parser.add_argument(
        "--percentile",
        type=float,
        default=0.95,
        help="Perentile to report for oracle and learned_objective metrics."
    )
    parser.add_argument(
        "--seed", default=42, type=int, help="Random seed. Default 42."
    )

    return parser.parse_args()


def sample(
    n: int,
    model: Optional[Union[Path, str]] = None,
    max_molecule_length: int = 109,
    device: str = "cpu",
    use_encoder: bool = False,
    use_train: bool = False,
) -> Sequence[Any]:
    """
    Generates n molecules according to a model generator network.
    Input:
        n: number of molecules to generate.
        model: path to the generative model checkpoint. If None, then n
            molecules from the test dataset will be sampled instead.
        max_molecule_length: maximum molecule length. Default is 109 which
            is the maximum molecule length contained within the training,
            validation, and test datasets.
        device: device. Default CPU.
        use_encoder: whether to use the encoder to generate sampled latent
            space points from encoding valid molecules from the test dataset.
        use_train: whether to sample molecules according to the training
            dataset instead of the test dataset.
    Returns:
        molecules: a generated batch of molecules using SELFIES representation.
        kld: KL divergence between the encoded latent space distribution and
            the normal distribution with mean 0 and variance 1.
    """
    device = get_device(device)

    if use_encoder or not model:
        vocab = load_vocab("./data/molecules/vocab.json")
        datamodule = SELFIESDataModule(
            vocab=vocab,
            batch_size=n,
            max_molecule_length=max_molecule_length
        )
        if not model:
            if use_train:
                molecules = one_hot_encodings_to_selfies(
                    next(iter(datamodule.train_dataloader())), vocab=vocab
                )
            else:
                molecules = one_hot_encodings_to_selfies(
                    next(iter(datamodule.test_dataloader())), vocab=vocab
                )
            return molecules, None

    vae = SELFIESVAEModule.load_from_checkpoint(
        model, map_location=device
    )
    vae.eval()
    molecules = []
    if use_encoder:
        if use_train:
            tokens = next(iter(datamodule.train_dataloader()))
        else:
            tokens = next(iter(datamodule.test_dataloader()))
        z, mu, log_var = vae.encoder(tokens.flatten(start_dim=1))
        z = torch.unsqueeze(z, dim=1)
        kld = -0.5 * torch.mean(1.0 + log_var - (mu * mu) - torch.exp(log_var))
        kld = kld.item()
    else:
        z = torch.randn((n, 1, vae.hparams.z_dim)).to(device)
        kld = None
    with torch.no_grad():
        for mol_z in tqdm(z, leave=False, desc="Generating Molecules"):
            mol = []
            mol_z = torch.unsqueeze(mol_z, dim=0)
            hidden = vae.decoder.init_hidden(batch_size=1)
            for _ in range(max_molecule_length):
                ote_output, hidden = vae.decoder(mol_z, hidden)
                ote_output = F.softmax(
                    torch.flatten(ote_output).detach(), dim=-1
                )
                mol.append(torch.argmax(ote_output, dim=0).data.tolist())
            mol = tokens_to_selfies(
                torch.tensor([mol]), vocab=vae.hparams.vocab
            )
            molecules.append(mol[0])
    return molecules, kld


def eval_molecules(
    molecules: Sequence[str],
    use_oracle: bool = False
) -> np.ndarray:
    """
    Evaluates the objective on a batch of molecules.
    Input:
        molecules: a generated batch of molecules using SELFIES representation.
        use_oracle: whether to use the oracle for objective calculation.
    """
    vocab = load_vocab("./data/molecules/vocab.json")
    if use_oracle:
        objective = logp_wrapper
    else:
        objective = SELFIESObjective(
            vocab=vocab, surrogate_ckpt="./MolOOD/checkpoints/regressor.ckpt"
        )
        objective.eval()
    vals = []
    for mol in molecules:
        if use_oracle:
            vals.append(objective(mol))
        else:
            mol = torch.tensor(
                [[vocab[tok] for tok in sf.split_selfies(mol)]],
                dtype=torch.int64
            )
            print(mol)
            vals.append(objective(mol).item())
    return vals


def plot_histograms(
    values: Dict[str, Sequence[float]],
    xrange: Optional[Sequence[float]] = None,
    savepath: Union[Path, str] = None
) -> None:
    """
    Plots oracle generated objective values versus neural net-generated values.
    Input:
        values: a dict mapping string labels to lists of objective values.
        xrange: range over values on the x-axis to plot the histogram.
        savepath: file path to save the plot. Default not saved.
    Returns:
        None.
    """
    colors = ["#E64B3588", "#4DBBD588", "#3C548888", "#00A08788", "#F39B7F"]
    labels = list(values.keys())
    objectives = [values[k] for k in labels]
    plt.figure(figsize=(10, 8))
    for obj, label, color in zip(objectives, labels, colors):
        plt.hist(obj, bins=100, label=label, color=color, range=xrange)
    plt.xlabel("Neural Network - Oracle Objective Value")
    plt.ylabel("Number of Molecules")
    plt.legend(bbox_to_anchor=(0.5, 1.25), loc="upper center")
    plt.tight_layout()
    if not savepath:
        plt.show()
    else:
        plt.savefig(savepath, transparent=True, bbox_inches="tight", dpi=600)


def selfies_to_fingerprints(
    selfies: Sequence[int]
) -> Sequence[DataStructs.cDataStructs.ExplicitBitVect]:
    """
    Converts a list of SELFIES molecule representations to a list of explicit
    bit vectors.
    Input:
        selfies: a list of SELFIES molecule representations.
    Returns:
        A corresponding list of explicit bit vector representations.
    """
    return [
        FingerprintMols.FingerprintMol(
            Chem.MolFromSmiles(sf.decoder(mol)),
            minPath=1,
            maxPath=7,
            fpSize=2048,
            bitsPerHash=2,
            useHs=True,
            tgtDensity=0.0,
            minSize=128
        )
        for mol in selfies
    ]


def main():
    args = build_args()
    seed_everything(seed=args.seed)
    plot_config(fontsize=16)

    xp, _ = sample(args.num_molecules, model=None, use_train=True)
    network_xp = eval_molecules(xp, use_oracle=False)
    oracle_xp = eval_molecules(xp, use_oracle=True)
    diff_xp = [x - y for x, y in zip(network_xp, oracle_xp)]

    xq_src, _ = sample(args.num_molecules, model=None, use_train=False)
    network_xq_src = eval_molecules(xq_src, use_oracle=False)
    oracle_xq_src = eval_molecules(xq_src, use_oracle=True)
    diff_xq_src = [x - y for x, y in zip(network_xq_src, oracle_xq_src)]

    xq_gen, _ = sample(args.num_molecules, model=args.model, use_encoder=False)
    network_xq_gen = eval_molecules(xq_gen, use_oracle=False)
    oracle_xq_gen = eval_molecules(xq_gen, use_oracle=True)
    diff_xq_gen = [x - y for x, y in zip(network_xq_gen, oracle_xq_gen)]

    vocab = load_vocab("./data/molecules/vocab.json")
    xp_vs_xq_src, xp_vs_xq_gen, xq_src_vs_xq_gen = 0.0, 0.0, 0.0

    if args.metric.lower() == "diversity":
        print("Metric: Diversity (Higher is More Diverse)")
        print("N =", args.num_molecules)
        print("Training Distribution:", len(set(xp)) / args.num_molecules)
        print("Test Distribution:", len(set(xq_src)) / args.num_molecules)
        print("Generated Distribution:", len(set(xq_gen)) / args.num_molecules)
    elif args.metric.lower() == "learned_objective":
        print("Metric: Learned Objective (Higher is Better)")
        print("N =", args.num_molecules)
        idx = round(min(max(args.percentile, 0.0), 1.0) * args.num_molecules)
        xp_idx = np.argsort(np.array(network_xp))[idx]
        print(
            "Training Distribution:",
            f"{network_xp[xp_idx]} (Using Oracle: {oracle_xp[xp_idx]})"
        )
        xq_src_idx = np.argsort(np.array(network_xq_src))[idx]
        print(
            "Test Distribution:",
            f"{network_xq_src[xq_src_idx]}",
            f"(Using Oracle: {oracle_xq_src[xq_src_idx]})"
        )
        xq_gen_idx = np.argsort(np.array(network_xq_gen))[idx]
        print(
            "Generated Distribution:",
            f"{network_xq_gen[xq_gen_idx]}",
            f"(Using Oracle: {oracle_xq_gen[xq_gen_idx]})"
        )
    elif args.metric.lower() == "oracle":
        print("Metric: Oracle (Higher is Better)")
        print("N =", args.num_molecules)
        idx = round(min(max(args.percentile, 0.0), 1.0) * args.num_molecules)
        xp_idx = np.argsort(np.array(oracle_xp))[idx]
        print(f"Training Distribution: {oracle_xp[xp_idx]}")
        xq_src_idx = np.argsort(np.array(oracle_xq_src))[idx]
        print(f"Test Distribution: {oracle_xq_src[xq_src_idx]}")
        xq_gen_idx = np.argsort(np.array(network_xq_gen))[idx]
        print(f"Generated Distribution: {oracle_xq_gen[xq_gen_idx]}")
    else:
        if args.metric.lower() == "tanimoto":
            xp = selfies_to_fingerprints(xp)
            xq_src = selfies_to_fingerprints(xq_src)
            xq_gen = selfies_to_fingerprints(xq_gen)
        for i in range(args.num_molecules):
            if args.metric.lower() == "tanimoto":
                xp_vs_xq_src += 0.5 * max(
                    DataStructs.BulkTanimotoSimilarity(xp[i], xq_src)
                )
                xp_vs_xq_src += 0.5 * max(
                    DataStructs.BulkTanimotoSimilarity(xq_src[i], xp)
                )
                xp_vs_xq_gen += 0.5 * max(
                    DataStructs.BulkTanimotoSimilarity(xp[i], xq_gen)
                )
                xp_vs_xq_gen += 0.5 * max(
                    DataStructs.BulkTanimotoSimilarity(xq_gen[i], xp)
                )
                xq_src_vs_xq_gen += 0.5 * max(
                    DataStructs.BulkTanimotoSimilarity(xq_src[i], xq_gen)
                )
                xq_src_vs_xq_gen += 0.5 * max(
                    DataStructs.BulkTanimotoSimilarity(xq_gen[i], xq_src)
                )
            elif args.metric.lower() == "levenshtein":
                x = [str(vocab[tok]) for tok in sf.split_selfies(xp[i])]
                x_src = [
                    str(vocab[tok]) for tok in sf.split_selfies(xq_src[i])
                ]
                x_gen = [
                    str(vocab[tok]) for tok in sf.split_selfies(xq_gen[i])
                ]
                xp_vs_xq_src += Levenshtein.ratio(x, x_src)
                xp_vs_xq_gen += Levenshtein.ratio(x, x_gen)
                xq_src_vs_xq_gen += Levenshtein.ratio(x_src, x_gen)
        xp_vs_xq_src /= args.num_molecules
        xp_vs_xq_gen /= args.num_molecules
        xq_src_vs_xq_gen /= args.num_molecules

        print("Metric:", args.metric.title(), "(Higher is More Similar)")
        print("N =", args.num_molecules)
        print("Training vs. Test Distribution:", xp_vs_xq_src)
        print("Training vs. Generated Distribution:", xp_vs_xq_gen)
        print("Test vs. Generated Distribution:", xq_src_vs_xq_gen)

    if not args.disable_plot:
        values = {
            "Training Distribution Molecules": diff_xp,
            "Test Distribution Molecules": diff_xq_src,
            "Generated Molecules": diff_xq_gen
        }
        plot_histograms(values, xrange=[-40, 120], savepath=args.savepath)


if __name__ == "__main__":
    main()
