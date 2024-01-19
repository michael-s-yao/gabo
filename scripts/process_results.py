"""
Experimental results parsing script and associated helper functions.

Author(s):
    Michael Yao

Licensed under the MIT License. Copyright University of Pennsylvania 2023.
"""
import argparse
import os
import sys
import numpy as np
from collections import defaultdict
from typing import NamedTuple

sys.path.append(".")
import mbo  # noqa
import design_bench
from design_bench.datasets.discrete.chembl_dataset import ChEMBLDataset
from models.oracle import BraninOracle


class Experiment(NamedTuple):
    id_: str
    method: str
    task_name: str
    args: str
    seed: int


def parse_subdir_name(subdir: str) -> Experiment:
    """
    Parses the name of a subdirectory of experimental results into the
    experiment's relevant properties. This function works for results and
    directories generated by the shell scripts in the `./scripts` sub-
    directory.
    Input:
        subdir: directory name as generated by a shell script from the `script`
            directory.
    Returns:
        An Experiment named tuple containing the relevant property values.
    """
    id_ = subdir.rsplit("-", 1)[0]
    method, subdir = subdir.split("-", 1)
    task_name = ""
    while design_bench.registration.TASK_PATTERN.search(task_name) is None:
        suffix, subdir = subdir.split("-", 1)
        task_name += "-" + suffix
    task_name = task_name[1:] if task_name.startswith("-") else task_name
    args = subdir.rsplit("-", 1)
    args, seed = ("", args) if len(args) == 1 else (args[0], args[-1])
    seed = int(seed[0] if isinstance(seed, list) else seed)
    return Experiment(id_, method, task_name, args, seed)


def build_args() -> argparse.Namespace:
    """
    Defines the arguments for the main parsing function.
    Input:
        None.
    Returns:
        A namespace containing the relevant argument values.
    """
    parser = argparse.ArgumentParser(
        description="COMBO-SCR Experimental Analysis"
    )
    parser.add_argument(
        "--top-k", type=int, required=True, help="Top k values to report."
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="./db-results",
        help="Path to the results directory. Default `./db-results`."
    )
    return parser.parse_args()


def main():
    args = build_args()

    results = defaultdict(lambda: [])
    ninf = -1e12
    for subdir in os.listdir(args.results_dir):
        exp = parse_subdir_name(subdir)

        designs = np.load(
            os.path.join(args.results_dir, subdir, "solution.npy")
        )
        preds = np.load(
            os.path.join(args.results_dir, subdir, "predictions.npy")
        )
        scores = np.load(
            os.path.join(args.results_dir, subdir, "scores.npy")
        )
        if exp.method == "ddom":
            designs = designs[-1, :, :]
            preds = preds[-1, :, :]
            scores = scores[-1, :, :]
        designs = designs.reshape(-1, designs.shape[-1])
        preds = preds.flatten()
        scores = scores.flatten()

        # Invalidate predictions for designs outside of the valid domain.
        if exp.task_name == os.environ["BRANIN_TASK"]:
            bounds = BraninOracle().oracle.bounds.detach().cpu().numpy()
            for dim in [0, 1]:
                preds = np.where(designs[:, dim] < bounds[0, dim], ninf, preds)
                preds = np.where(designs[:, dim] > bounds[1, dim], ninf, preds)
        elif exp.task_name == os.environ["MNIST_TASK"]:
            preds = np.where(
                np.any(designs[:, dim] < 0.0, axis=-1), ninf, preds
            )
            preds = np.where(
                np.any(designs[:, dim] > 1.0, axis=-1), ninf, preds
            )
        elif exp.task_name.endswith(os.environ["CHEMBL_TASK"]):
            _, standard_type, assay_chembl_id, _ = exp.task_name.split("_")
            y_shards = ChEMBLDataset.register_y_shards(
                assay_chembl_id=assay_chembl_id, standard_type=standard_type
            )
            y = np.vstack([np.load(shard.disk_target) for shard in y_shards])
            scores = (scores - y.min()) / (y.max() - y.min())

        idxs = np.argsort(preds)
        results[exp.id_] += scores[idxs[-args.top_k:]].tolist()

    results = {
        key: f"{np.mean(val)} +/- {np.std(val)}"
        for key, val in results.items()
    }
    for key in sorted(list(results.keys())):
        print(key, results[key])


if __name__ == "__main__":
    main()
