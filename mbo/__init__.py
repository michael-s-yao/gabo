"""
Registers custom model-based optimization (MBO) tasks with design-bench.

Author(s):
    Michael Yao @michael-s-yao

Citation(s):
    [1] Trabucco B, Kumar A, Geng X, Levine S. Design-Bench: Benchmarks for
        data-driven offline model-based optimization. Proc ICML 162: 21658-76.
        (2022). https://proceedings.mlr.press/v162/trabucco22a.html

Licensed under the MIT License. Copyright University of Pennsylvania 2023.
"""
import os
import numpy as np
import selfies as sf
import sys
import logging
import torch
import transformers
import gpytorch
import warnings
from typing import Callable, Union

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
sys.path.append(".")
sys.path.append("design-baselines")
torch.set_default_dtype(torch.float64)
transformers.logging.set_verbosity_error()
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings(
    "ignore", category=gpytorch.utils.warnings.NumericalWarning
)
import design_bench
import data
from design_bench.datasets.continuous_dataset import ContinuousDataset
from design_bench.datasets.discrete_dataset import DiscreteDataset
from models.features import SELFIESMorganFingerprintFeatures
from models.oracle import (
    BraninOracle, MNISTOracle, MoleculeOracle, WarfarinDosingOracle
)


TASK_DATASETS = {
    os.environ["BRANIN_TASK"]: data.BraninDataset,
    os.environ["MNIST_TASK"]: data.MNISTIntensityDataset,
    os.environ["MOLECULE_TASK"]: data.PenalizedLogPDataset,
    os.environ["CHEMBL_TASK"]: data.SELFIESChEMBLDataset,
    os.environ["WARFARIN_TASK"]: data.WarfarinDosingDataset
}


class OracleWrapper:
    """Wrapper class for the oracle function for MBO tasks."""

    def __init__(
        self,
        task_name: str,
        dataset: Union[ContinuousDataset, DiscreteDataset]
    ):
        """
        Args:
            task_name: the name of the model-based optimization (MBO) task.
            dataset: the dataset associated with the MBO task.
        """
        self.task_name = task_name
        self.dataset = dataset
        self.oracle = self.task_oracle()

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the value of the oracle on the input design x.
        Input:
            x: the input batch of designs.
        Returns:
            The values of the oracle for each input design.
        """
        if self.task_name == os.environ["MOLECULE_TASK"]:
            if issubclass(x.dtype.type, np.floating):
                x = self.dataset.to_integers(x)
            x = x[np.newaxis, ...] if x.ndim == 1 else x
            y = np.array([
                [self.oracle(sf.decoder(self.dataset.data.decode(tok)))]
                for tok in x.astype(np.int32)
            ])
            return y.astype(np.float32)
        elif self.task_name == os.environ["BRANIN_TASK"]:
            y = self.oracle(torch.from_numpy(x)).detach().cpu().numpy()
            return y[:, np.newaxis]
        return self.oracle(x)[:, np.newaxis]

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the value of the oracle on the input design x.
        Input:
            x: the input batch of designs.
        Returns:
            The values of the oracle for each input design.
        """
        return self(x)

    def task_oracle(self) -> Callable:
        """
        Returns the oracle function associated with the MBO task.
        Input:
            None.
        Returns:
            The oracle function associated with the task name.
        """
        if self.task_name == os.environ["BRANIN_TASK"]:
            return BraninOracle()
        elif self.task_name == os.environ["MNIST_TASK"]:
            return MNISTOracle()
        elif self.task_name == os.environ["MOLECULE_TASK"]:
            return MoleculeOracle("logP")
        elif self.task_name == os.environ["WARFARIN_TASK"]:
            return WarfarinDosingOracle(
                self.dataset.transform,
                column_names=self.dataset.column_names,
                mean_dose=self.dataset.mean_dose,
                use_pharmacogenetic_algorithm=True
            )


def register(task_name: str) -> None:
    """
    Register a specification for a model-based optimization task.
    Input:
        task_name: the name of the model-based optimization task.
        verbose: whether to print verbose outputs. Default True.
    Returns:
        None.
    """
    dataset_kwargs = {
        "max_samples": None,
        "distribution": None,
        "max_percentile": 100.0,
        "min_percentile": 0.0
    }
    oracle_kwargs = {}
    oracle = lambda dataset: OracleWrapper(task_name, dataset)  # noqa
    if task_name == os.environ["BRANIN_TASK"]:
        dataset_kwargs["max_percentile"] = 80.0
    elif task_name == os.environ["MNIST_TASK"]:
        dataset_kwargs["root"] = "./data/mnist"
    elif task_name == os.environ["MOLECULE_TASK"]:
        dataset_kwargs["fname"] = "./data/molecules/val_selfie.gz"
    elif task_name == os.environ["CHEMBL_TASK"]:
        # Use the same dataset kwargs as in the original ChEMBL task
        # specification.
        dataset_kwargs["max_percentile"] = 40.0
        dataset_kwargs.update({
            "assay_chembl_id": "CHEMBL3885882", "standard_type": "MCHC"
        })
        # Use the same oracle kwargs as in the original ChEMBL task
        # specification except with the SELFIESMorganFingerprintFeatures
        # feature extractor instead.
        oracle_kwargs = {
            "noise_std": 0.0,
            "max_samples": 2000,
            "distribution": None,
            "max_percentile": 100,
            "min_percentile": 0,
            "override_input_spec": True,
            "feature_extractor": SELFIESMorganFingerprintFeatures(
                dtype=np.int32
            ),
            "model_kwargs": {
                "n_estimators": 100, "max_depth": 100, "max_features": "auto"
            },
            "split_kwargs": {
                "val_fraction": 0.5,
                "subset": None,
                "shard_size": 50_000,
                "to_disk": False
            }
        }
        oracle = "design_bench.oracles.sklearn:RandomForestOracle"
    elif task_name == os.environ["WARFARIN_TASK"]:
        dataset_kwargs.update({
            "dir_name": "./data/warfarin",
            "normalize_y": True,
            "thresh_min_y": -10.0
        })

    design_bench.register(
        task_name,
        TASK_DATASETS[task_name],
        oracle=oracle,
        dataset_kwargs=dataset_kwargs,
        oracle_kwargs=oracle_kwargs
    )
    logging.info(f"Registered task {task_name}.")
    return


for task_name in TASK_DATASETS.keys():
    register(task_name)
