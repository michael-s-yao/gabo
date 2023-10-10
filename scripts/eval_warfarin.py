"""
Warfarin conterfactual generation model evaluation script and helper functions.

Author(s):
    Michael Yao

Citation(s):
    [1] Goncalves A, Ray P, Soper B, Stevens J, Coyle L, Sales AP. Generation
        and evaluation of synthetic patient data. BMC Med Res Methodology.
        20:108. (2020). https://doi.org/10.1186/s12874-020-00977-1
    [2] Woo M, Reiter JP, Oganian A, Karr AF. Global measures of data utility
        for microdata masked for disclosure limitation. J Privacy and
        Confidentiality 1(1). (2009). https://doi.org/10.29012/jpc.v1i1.568

Licensed under the MIT License. Copyright University of Pennsylvania 2023.
"""
import argparse
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import sys
from pathlib import Path
from scipy.stats import ttest_ind
from sklearn.cluster import KMeans
import torch
from typing import Dict, Optional, Sequence, Tuple, Union

sys.path.append(".")
from data.iwpc import IWPCWarfarinDataModule
from models.ctgan import CTGANLightningModule
from models.mortality_estimator import WarfarinMortalityLightningModule
from experiment.utility import seed_everything


def build_args() -> argparse.Namespace:
    """
    Builds the arguments for the model evaluation script.
    Input:
        None.
    Returns:
        The argument values for the script.
    """
    parser = argparse.ArgumentParser(description="CTGAN Model Evaluation")
    parser.add_argument(
        "--ckpt", type=str, required=True, help="Model checkpoint to evaluate."
    )
    parser.add_argument(
        "--savepath",
        type=str,
        default=None,
        help="Path to save the warfarin-associated cost estimation plot to."
    )
    parser.add_argument(
        "--batch_size", type=int, default=128, help="Batch size. Default 128."
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed. Default 42."
    )
    return parser.parse_args()


def parse_features(
    X: torch.Tensor, X_attributes: Sequence[str]
) -> Tuple[torch.Tensor, Sequence[str]]:
    """
    Processes model outputs and original raw data into vectors that can
    be analyzed.
    Input:
        X: A BxF matrix of B observations of F features.
        X_attributes: a list of the feature names.
    Returns:
        X: the matrix of the parsed features.
        trimmed_attributes: a list of the parsed feature names.
    """
    attributes = defaultdict(lambda: [])
    for attr in X_attributes:
        key = attr.split("_")[0].replace(".normalized", "").replace(
            ".component", ""
        )
        attributes[key].append(attr)
    trimmed_attributes = []
    for attr, lst in attributes.items():
        if attr.startswith("Height (cm)") or attr.startswith("Weight (kg)"):
            trimmed_attributes += [attr]
        elif len(lst) > 2 or attr.startswith("Gender"):
            trimmed_attributes += lst
        else:
            trimmed_attributes += [attr]
    if X.size(dim=-1) != len(trimmed_attributes):
        error_message = f"X dimensions {X.size()} and number of attributes "
        error_message += f"{len(trimmed_attributes)} do not match!"
        raise ValueError(error_message)
    return X, trimmed_attributes


def divergence(
    P: np.ndarray,
    Q: np.ndarray,
    attributes: Sequence[str],
    continuous: Sequence[str] = [
        "Therapeutic Dose of Warfarin",
        "Height (cm)",
        "Weight (kg)"
    ],
    labels: Optional[Sequence[str]] = None,
    compute_jsd: bool = True,
    verbose: bool = True
) -> Dict[str, float]:
    """
    Calculates the JS or KL Divergence between two sets of observations P and
    Q by variable.
    Input:
        P: A BxF matrix of B observations of F features.
        Q: A CxF matrix of C observations of F features.
        attributes: a list of the feature names.
        continuous: a list of the continuous variable attributes.
        labels: optional string labels of P and Q distributions.
        compute_jsd: whether to compute the JS Divergence instead of the KL
            divergence. Default True.
        verbose: whether to print calculated metrics and information.
    Returns:
        The Frobenius norm of the difference between the Pearson correlation
        matrices of P and Q.
    """
    if labels and verbose:
        p_label, q_label = labels
        print(f"\nP {p_label} (N = {len(P)}) vs. Q {q_label} (N = {len(Q)})")
    feature_idxs = defaultdict(lambda: [])
    for attr in attributes:
        key = attr.split("_")[0].replace(".normalized", "").replace(
            ".component", ""
        )
        feature_idxs[key].append(attributes.index(attr))
    div = {}
    for attr, idxs in feature_idxs.items():
        if len(idxs) == 1:
            p, q = P[:, idxs[0]], Q[:, idxs[0]]
            if np.all(np.logical_and(p <= 1.0, p >= 0.0)):
                p, q = p > 0.5, q > 0.5
        else:
            p, q = P[:, min(idxs):max(idxs)], Q[:, min(idxs):max(idxs)]
            p, q = np.argmax(p, axis=-1), np.argmax(q, axis=-1)
        if attr in continuous:
            if not verbose:
                continue
            print(
                f"  {attr}:",
                f"(P) {np.mean(p):.5f} " + u"\u00b1" + f" {np.std(p):.5f};",
                f"(Q) {np.mean(q):.5f} " + u"\u00b1" + f" {np.std(q):.5f}"
            )
            equal_var = 0.25 <= np.var(p) / np.var(q) <= 4.0
            print(f"  p = {ttest_ind(p, q, equal_var=equal_var).pvalue:.5f}")
            continue
        vals = sorted(list(set(p.tolist()) & set(q.tolist())))
        p = np.array([np.sum(p == x) / p.size for x in vals])
        q = np.array([np.sum(q == x) / q.size for x in vals])
        if compute_jsd:
            m = 0.5 * (p + q)
            div[attr] = 0.5 * (
                np.sum(p * np.log2(p / m)) + np.sum(q * np.log2(q / m))
            )
        else:
            div[attr] = np.sum(p * np.log2(p / q))
        if verbose:
            print(f"  {attr}: {div[attr]:.5f}")
            print(f"  (P): {p}")
            print(f"  (Q): {q}")
    avg_divergence = sum([div[attr] for attr in div.keys()]) / len(div.keys())
    print(f" Average Divergence: {avg_divergence}")
    return div


def pcd(
    P: np.ndarray,
    Q: np.ndarray,
    labels: Optional[Sequence[str]] = None,
    verbose: bool = True,
    seed: int = 42,
    max_eps: float = 1e-12
) -> float:
    """
    Calculates the pairwise correlation difference (PCD) between two matrices
    of observations P and Q.
    Input:
        P: A BxF matrix of B observations of F features.
        Q: A CxF matrix of C observations of F features.
        labels: optional string labels of P and Q distributions.
        verbose: whether to print calculated metrics and information.
        seed: random seed. Default 42.
        max_eps: maximum perturbation to the matrix elements.
    Returns:
        The Frobenius norm of the difference between the Pearson correlation
        matrices of P and Q.
    """
    if labels and verbose:
        p_label, q_label = labels
        print(f"\nP {p_label} (N = {len(P)}) vs. Q {q_label} (N = {len(Q)})")
    rng = np.random.RandomState(seed=seed)
    p_eps = max_eps * rng.normal(size=P.shape)
    q_eps = max_eps * rng.normal(size=Q.shape)
    p_corr = np.corrcoef(P + p_eps, rowvar=False)
    q_corr = np.corrcoef(Q + q_eps, rowvar=False)
    pcd = np.linalg.norm(p_corr - q_corr, ord="fro")
    if verbose:
        print(f"  PCD Metric: {pcd:.5f}")
    return pcd


def negative_log_cluster(
    P: np.ndarray,
    Q: np.ndarray,
    labels: Optional[Sequence[str]] = None,
    G: int = 5,
    verbose: bool = True
) -> float:
    """
    Calculates the negative log cluster metric between two sets of
    observations P and Q.
    Input:
        P: A BxF matrix of B observations of F features.
        Q: A CxF matrix of C observations of F features.
        labels: optional string labels of P and Q distributions.
        G: number of clusters. Default 5.
        verbose: whether to print calculated metrics and information.
    Returns:
        The Frobenius norm of the difference between the Pearson correlation
        matrices of P and Q.
    """
    if labels and verbose:
        p_label, q_label = labels
        print(f"\nP {p_label} (N = {len(P)}) vs. Q {q_label} (N = {len(Q)})")
    merged = np.concatenate((P, Q), axis=0)
    true_labels = np.concatenate(
        (np.ones(P.shape[0]), np.zeros(Q.shape[0])), axis=0
    )
    kmeans = KMeans(n_clusters=G, n_init="auto")
    kmeans.fit(merged)
    cluster_labels = np.array(kmeans.labels_)
    arg = 0.0
    for k in range(G):
        count = np.sum(np.where(cluster_labels == k, 1.0, 0.0) * true_labels)
        prop = count / np.sum(np.where(cluster_labels == k, 1.0, 0.0))
        c = P.shape[0] / (P.shape[0] + Q.shape[0])
        arg += ((prop - c) ** 2) / G
    res = -np.log(arg)
    if verbose:
        print(f"  Log Cluster Metric: {res:.5f}")
    return res


def support_coverage(
    P: np.ndarray,
    Q: np.ndarray,
    attributes: Sequence[str],
    continuous: Sequence[str] = [
        "Therapeutic Dose of Warfarin",
        "Height (cm)",
        "Weight (kg)"
    ],
    labels: Optional[Sequence[str]] = None,
    verbose: bool = True
) -> float:
    """
    Calculates the average support coverage between two sets of observations P
    and Q.
    Input:
        P: A BxF matrix of B observations of F features.
        Q: A CxF matrix of C observations of F features.
        attributes: a list of the feature names.
        continuous: a list of the continuous variable attributes.
        labels: optional string labels of P and Q distributions.
        verbose: whether to print calculated metrics and information.
    Returns:
        The Frobenius norm of the difference between the Pearson correlation
        matrices of P and Q.
    """
    if labels and verbose:
        p_label, q_label = labels
        print(f"\nP {p_label} (N = {len(P)}) vs. Q {q_label} (N = {len(Q)})")
    feature_idxs = defaultdict(lambda: [])
    for attr in attributes:
        key = attr.split("_")[0].replace(".normalized", "").replace(
            ".component", ""
        )
        feature_idxs[key].append(attributes.index(attr))
    coverage = 0.0
    for attr, idxs in feature_idxs.items():
        if len(idxs) == 1:
            p, q = P[:, idxs[0]], Q[:, idxs[0]]
            if np.all(np.logical_and(p <= 1.0, p >= 0.0)):
                p, q = p > 0.5, q > 0.5
        else:
            p, q = P[:, min(idxs):max(idxs)], Q[:, min(idxs):max(idxs)]
            p, q = np.argmax(p, axis=-1), np.argmax(q, axis=-1)
        if attr in continuous:
            rv = np.max(p) - np.min(p)
            sv = min(np.max(p), np.max(q)) - max(np.min(p), np.min(q))
        else:
            rv = len(set(p.tolist()))
            sv = len(set(q.tolist()) & set(p.tolist()))
        coverage += (sv / rv) / len(feature_idxs.keys())
    if verbose:
        print(f"  Support Coverage: {coverage:.5f}")
    return coverage


def estimate_mortality_cost(
    patients: torch.Tensor,
    ckpt: Union[Path, str] = "./ckpts/warfarin_cost_estimator.ckpt"
) -> torch.Tensor:
    """
    Estimates the mortality cost for the given patient data using a learned
    warfarin-associated mortality cost estimation model.
    Input:
        patients: a BxF tensor of F-dimensional features of B patients.
        ckpt: checkpoint to the trained Warfarin cost estimator function.
    Returns:
        A vector of B estimations for the mortality cost.
    """
    cost = WarfarinMortalityLightningModule.load_from_checkpoint(
        ckpt, map_device=patients.device
    )
    cost.eval()
    cost = cost.to(patients.device)
    with torch.no_grad():
        return cost(patients)


def main():
    args = build_args()
    seed_everything(seed=args.seed)

    datamodule = IWPCWarfarinDataModule(
        batch_size=args.batch_size, seed=args.seed
    )
    datamodule.prepare_data()
    datamodule.setup()

    model = CTGANLightningModule.load_from_checkpoint(
        args.ckpt, map_location=torch.device("cpu")
    )

    train_batches, curr = [], []
    for i in range(len(datamodule.train)):
        pt = datamodule.train[i]
        if len(curr) < args.batch_size:
            curr.append(pt)
            continue
        train_batches.append([pt_ for pt_ in curr])
        curr = [pt]

    test_batches, curr = [], []
    for i in range(len(datamodule.test)):
        pt = datamodule.test[i]
        if len(curr) < args.batch_size:
            curr.append(pt)
            continue
        test_batches.append([pt_ for pt_ in curr])
        curr = [pt]

    train, train_generated = None, None
    for batch in train_batches:
        batch = datamodule.collate_fn(batch)
        Xp, attributes = parse_features(
            datamodule.invert(batch.X, batch.X_attributes), batch.X_attributes
        )
        # Sample from the model.
        Xq = model._activate(model(batch), batch.X_attributes)
        Xq, _ = parse_features(
            datamodule.invert(Xq, batch.X_attributes), batch.X_attributes
        )
        if train is None:
            train, train_generated = Xp, Xq
        else:
            train = torch.cat((train, Xp), dim=0)
            train_generated = torch.cat((train_generated, Xq), dim=0)

    test, test_generated = None, None
    for batch in test_batches:
        batch = datamodule.collate_fn(batch)
        Xp, attributes = parse_features(
            datamodule.invert(batch.X, batch.X_attributes), batch.X_attributes
        )
        # Sample from the model.
        Xq = model._activate(model(batch), batch.X_attributes)
        Xq, _ = parse_features(
            datamodule.invert(Xq, batch.X_attributes), batch.X_attributes
        )
        if test is None:
            test, test_generated = Xp, Xq
        else:
            test = torch.cat((test, Xp), dim=0)
            test_generated = torch.cat((test_generated, Xq), dim=0)

    # Warfarin-associated mortality cost estimation.
    print("\nEstimated Mortality Cost")
    train_cost = estimate_mortality_cost(train)
    test_cost = estimate_mortality_cost(test)
    generated_cost = estimate_mortality_cost(train_generated)

    plot = False
    if plot:
        plt.figure(figsize=(10, 6))
        plt.hist(train_cost, bins=100, density=True, alpha=0.6, label="Training")
        plt.hist(test_cost, bins=100, density=True, alpha=0.5, label="Test")
        plt.hist(
            generated_cost, bins=100, density=True, alpha=0.4, label="Generated"
        )
        plt.xlabel("Estimated Warfarin-Associated Mortality Cost")
        plt.ylabel("Density")
        plt.legend()
        if args.savepath is None:
            plt.show()
        else:
            print(f"Saved estimated mortality cost plot to {args.savepath}")
            plt.savefig(
                args.savepath, transparent=True, dpi=600, bbox_inches="tight"
            )
        plt.close()

    train, test = train.detach().cpu().numpy(), test.detach().cpu().numpy()
    train_generated = train_generated.detach().cpu().numpy()
    test_generated = test_generated.detach().cpu().numpy()

    # JS Divergence (Lower is Better). Goncalves et al. [1].
    print("\nJS Divergence (Lower is Better)")
    divergence(test, train, attributes, labels=["Test", "Train"])
    divergence(test, test_generated, attributes, labels=["Test", "Generated"])

    # Pairwise Correlation Difference (Lower is Better). Goncalves et al. [1].
    print("\nPairwise Correlation Difference (Lower is Better)")
    pcd(test, train, seed=args.seed, labels=["Test", "Train"])
    pcd(test, test_generated, seed=args.seed, labels=["Test", "Generated"])

    # Negative Log Cluster (Higher is Better).
    # Goncalves et al. [1] and Woo et al. [2].
    print("\nNegative Log Cluster (Higher is Better)")
    negative_log_cluster(test, train, labels=["Test", "Train"])
    negative_log_cluster(test, test_generated, labels=["Test", "Generated"])

    # Support Coverage (Higher is Better). Goncalves et al. [1].
    print("\nSupport Coverage (Higher is Better)")
    support_coverage(test, train, attributes, labels=["Test", "Train"])
    support_coverage(
        test, test_generated, attributes, labels=["Test", "Generated"]
    )


if __name__ == "__main__":
    main()
