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
from typing import Any, Dict, Optional, Sequence, Tuple, Union

sys.path.append(".")
sys.path.append("MolOOD")
from data.molecule import (
    SELFIESDataModule,
    tokens_to_selfies,
    one_hot_encodings_to_selfies
)
from fcd_torch import FCD
from models.objective import SELFIESObjective
from models.vae import SELFIESVAEModule
from models.seqgan import MolGANModule
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
    metric_choices = [
        "tanimoto",
        "levenshtein",
        "diversity",
        "learned_objective",
        "oracle",
        "recon"
    ]
    parser.add_argument(
        "--metric",
        type=str,
        nargs="+",
        choices=metric_choices,
        default=metric_choices,
        help="Metric for molecule similarity. Default all metrics evaluated."
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


def sample_MolGAN(
    n: int,
    model: Optional[Union[Path, str]] = None,
    max_molecule_length: int = 109,
    device: str = "cpu"
) -> Sequence[Any]:
    """
    Generates n molecules according to a MolGAN generator network.
    Input:
        n: number of molecules to generate.
        model: path to the generative model checkpoint. If None, then n
            molecules from the test dataset will be sampled instead.
        max_molecule_length: maximum molecule length. Default is 109 which
            is the maximum molecule length contained within the training,
            validation, and test datasets.
        device: device. Default CPU.
    Returns:
        molecules: a generated batch of molecules using SELFIES representation.
    """
    device = get_device(device)
    vocab = load_vocab("./data/molecules/vocab.json")

    if not model:
        datamodule = SELFIESDataModule(
            vocab=vocab,
            batch_size=n,
            max_molecule_length=max_molecule_length
        )
        molecules = one_hot_encodings_to_selfies(
            next(iter(datamodule.test_dataloader())), vocab=vocab
        )
        return molecules

    gan = MolGANModule.load_from_checkpoint(model, map_location=device)
    gan.eval()
    dummy = torch.zeros((n, gan.hparams.max_molecule_length))
    with torch.no_grad():
        output = gan(dummy)
    return tokens_to_selfies(output, vocab)


def recon_accuracy(
    model: Union[Path, str],
    device: str = "cpu",
    max_molecule_length: int = 109
) -> float:
    """
    Computes the token-wise reconstruction accuracy of a generator network.
    Input:
        model: path to the generative model checkpoint.
        device: device. Default CPU.
        max_molecule_length: maximum molecule length. Default is 109 which
            is the maximum molecule length contained within the training,
            validation, and test datasets.
    Returns:
        Average recon accuracy of the specified model on the test dataset.
    """
    device = get_device(device)
    vocab = load_vocab("./data/molecules/vocab.json")
    datamodule = SELFIESDataModule(
        vocab=vocab,
        batch_size=256,
        max_molecule_length=max_molecule_length
    )
    vae = SELFIESVAEModule.load_from_checkpoint(
        model, map_location=device
    )
    vae.eval()
    total, recon_acc = 0, []
    with torch.no_grad():
        for tokens in tqdm(
            iter(datamodule.test_dataloader()),
            desc="Computing Recon Accuracy",
            leave=False
        ):
            n_molecules, _, _ = tokens.size()
            total += n_molecules
            recons, _, _ = vae(tokens)
            compare = torch.eq(
                torch.argmax(tokens, dim=-1), torch.argmax(recons, dim=-1)
            )
            recon_acc.append(
                n_molecules * torch.sum(compare) / torch.numel(compare)
            )
    return sum(recon_acc) / total


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
    suffix = "Oracle" if use_oracle else "Learned Objective"
    desc = f"Evaluating Molecules Using {suffix}"
    for mol in tqdm(molecules, desc=desc, leave=False):
        if use_oracle:
            vals.append(objective(mol))
        else:
            mol = torch.tensor(
                [[vocab[tok] for tok in sf.split_selfies(mol)]],
                dtype=torch.int64
            )
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


def source_all_metrics(
    partition: str = "train",
    vocab_path: Union[Path, str] = "./data/molecules/vocab.json",
    max_molecule_length: int = 109,
) -> None:
    """
    Evaluates the `diversity`, `learned_objective`, and `oracle` metrics on
    the entirety of a source dataset partition.
    Input:
        partition: the source dataset partition to evaluate the metrics on.
            One of [`train`, `test`]. Default `train`.
        vocab_path: file path to vocab dict.
        max_molecule_length: maximum molecule length. Default 109.
    Returns:
        None. All metrics are printed to `stdout`.
    """
    vocab = load_vocab(vocab_path)
    datamodule = SELFIESDataModule(
        vocab=vocab,
        batch_size=4,
        max_molecule_length=max_molecule_length
    )
    if partition.lower() == "train":
        dataset = iter(datamodule.train_dataloader())
    elif partition.lower() == "test":
        dataset = iter(datamodule.test_dataloader())
    else:
        raise ValueError(f"Unrecognized source data partition {partition}.")

    molecules = [0] * len(dataset)
    for i, mol in enumerate(
        tqdm(dataset, desc="Encoding Dataset", leave=False)
    ):
        molecules[i] = one_hot_encodings_to_selfies(mol, vocab=vocab)[0]

    network = eval_molecules(molecules, use_oracle=False)
    oracle = eval_molecules(molecules, use_oracle=True)

    print(f"{partition.title()} Dataset, N = {len(molecules)}")

    print(f"  Diversity: {len(set(molecules)) / len(molecules)}")
    S_network, S_oracle = S(network, oracle)
    print(
        "  Learned Objective (Higher is Better):",
        f"{S_network} (Using Oracle: {S_oracle})"
    )
    S_oracle, S_network = S(oracle, network)
    print(
        "  Oracle (Higher is Better):",
        f"{S_oracle} (Using Network: {S_network})"
    )


def S(
    objective_scores: Sequence[float],
    alternative_scores: Optional[Sequence[float]] = None
) -> Union[float, Tuple[float]]:
    """
    Computes the benchmark score S as a combination of the top-1, top-10, and
    top-100 objective scores as defined in Eq. (4) from Brown et al. (2019).
    Inputs:
        objective_scores: a list of molecule scores.
        alternative_scores: a list of molecules scores according to some other
            metric. Using the same top-1, top-10, and top-100 molecules
            according to the `objective_scores` input, S is also computed
            according to the alternative scores if provided.
    Returns:
        The corresponding benchmark score S.
    Citation(s):
        [1] Brown N, Fiscato M, Segler MHS, Vaucher AC. GuacaMol: Benchmarking
            models for de novo molecular design. J Chem Inf Model 59(3):1096-
            108. (2019). https://doi.org/10.1021/acs.jcim.8b00839
    """
    if not isinstance(objective_scores, np.ndarray):
        objective_scores = np.array(objective_scores)
    idxs = np.argsort(objective_scores)
    top_k = lambda k: np.mean(objective_scores[idxs[k:]])  # noqa
    s = (top_k(1) + top_k(10) + top_k(100)) / 3.0
    if alternative_scores is None:
        return s
    if not isinstance(alternative_scores, np.ndarray):
        alternative_scores = np.array(alternative_scores)
    top_k_alt = lambda k: np.mean(alternative_scores[idxs[k:]])  # noqa
    return s, (top_k_alt(1) + top_k_alt(10) + top_k_alt(100)) / 3.0


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

    print("N =", args.num_molecules)
    if "diversity" in args.metric:
        print("Metric: Diversity (Higher is More Diverse)")
        print("  Training Distribution:", len(set(xp)) / args.num_molecules)
        print("  Test Distribution:", len(set(xq_src)) / args.num_molecules)
        print(
            "  Generated Distribution:", len(set(xq_gen)) / args.num_molecules
        )
    if "learned_objective" in args.metric:
        S_train_network, S_train_oracle = S(network_xp, oracle_xp)
        S_test_network, S_test_oracle = S(network_xq_src, oracle_xq_src)
        S_gen_network, S_gen_oracle = S(network_xq_gen, oracle_xq_gen)
        print("Metric: Learned Objective (Higher is Better)")
        print(
            "  Training Distribution:",
            f"{S_train_network} (Using Oracle: {S_train_oracle})"
        )
        print(
            "  Test Distribution:",
            f"{S_test_network} (Using Oracle: {S_test_oracle})"
        )
        print(
            "  Generated Distribution:",
            f"{S_gen_network} (Using Oracle: {S_gen_oracle})"
        )
    if "oracle" in args.metric:
        S_train_oracle, S_train_network = S(oracle_xp, network_xp)
        S_test_oracle, S_test_network = S(oracle_xq_src, network_xq_src)
        S_gen_oracle, S_gen_network = S(oracle_xq_gen, network_xq_gen)
        print("Metric: Oracle (Higher is Better)")
        print(
            "  Training Distribution:",
            f"{S_train_oracle} (Using Network: {S_train_network})"
        )
        print(
            "  Test Distribution:",
            f"{S_test_oracle} (Using Network: {S_test_network})"
        )
        print(
            "  Generated Distribution:",
            f"{S_gen_oracle} (Using Network: {S_gen_network})"
        )
    if "fcd" in args.metric:
        fcd = FCD(device=get_device())
        xp_smiles = [sf.decoder(mol) for mol in xp]
        xq_src_smiles = [sf.decoder(mol) for mol in xq_src]
        xq_gen_smiles = [sf.decoder(mol) for mol in xq_gen]
        xp_vs_xq_src = fcd(xp_smiles, xq_src_smiles)
        xp_vs_xq_gen = fcd(xp_smiles, xq_gen_smiles)
        xq_src_vs_xq_gen = fcd(xq_src_smiles, xq_gen_smiles)
        print("Metric: FCD (Lower is More Similar)")
        print("  Training vs. Test Distribution:", xp_vs_xq_src)
        print("  Training vs. Generated Distribution:", xp_vs_xq_gen)
        print("  Test vs. Generated Distribution:", xq_src_vs_xq_gen)
    if "recon" in args.metric:
        print("Metric: Reconstruction Accuracy (Higher is Better)")
        print(f"  {args.model}: {recon_accuracy(model=args.model)}")
    if "levenshtein" in args.metric:
        for i in range(args.num_molecules):
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

        print("Metric: Levenshtein (Higher is More Similar)")
        print("  Training vs. Test Distribution:", xp_vs_xq_src)
        print("  Training vs. Generated Distribution:", xp_vs_xq_gen)
        print("  Test vs. Generated Distribution:", xq_src_vs_xq_gen)
    if "tanimoto" in args.metric:
        xp = selfies_to_fingerprints(xp)
        xq_src = selfies_to_fingerprints(xq_src)
        xq_gen = selfies_to_fingerprints(xq_gen)
        for i in range(args.num_molecules):
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
        xp_vs_xq_src /= args.num_molecules
        xp_vs_xq_gen /= args.num_molecules
        xq_src_vs_xq_gen /= args.num_molecules

        print("Metric: Tanimoto (Higher is More Similar)")
        print("  N =", args.num_molecules)
        print("  Training vs. Test Distribution:", xp_vs_xq_src)
        print("  Training vs. Generated Distribution:", xp_vs_xq_gen)
        print("  Test vs. Generated Distribution:", xq_src_vs_xq_gen)

    if not args.disable_plot:
        values = {
            "Training Distribution Molecules": diff_xp,
            "Test Distribution Molecules": diff_xq_src,
            "Generated Molecules": diff_xq_gen
        }
        plot_histograms(values, savepath=args.savepath)


if __name__ == "__main__":
    main()
