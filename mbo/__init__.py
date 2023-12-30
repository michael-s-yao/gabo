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
import warnings
from botorch.test_functions.synthetic import Branin
from typing import Callable, Dict, Union

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
sys.path.append(".")
torch.set_default_dtype(torch.float64)
transformers.logging.set_verbosity_error()
warnings.filterwarnings("ignore", category=FutureWarning)
import design_bench
from design_bench.datasets.continuous_dataset import ContinuousDataset
from design_bench.datasets.discrete_dataset import DiscreteDataset
from data.data import (
    BraninDataset, MNISTIntensityDataset, PenalizedLogPDataset
)
from molecules.utils import MoleculeObjective


BRANIN_TASK = "Branin-Branin-v0"


MNIST_TASK = "MNISTIntensity-L2-v0"


MOLECULE_TASK = "PenalizedLogP-Guacamol-v0"


TASK_DATASETS = {
    BRANIN_TASK: BraninDataset,
    MNIST_TASK: MNISTIntensityDataset,
    MOLECULE_TASK: PenalizedLogPDataset,
}


class OracleWrapper:
    """Wrapper class for the oracle function for MBO tasks."""

    def __init__(
        self, task: str, dataset: Union[ContinuousDataset, DiscreteDataset]
    ):
        """
        Args:
            task: the name of the model-based optimization (MBO) task.
            dataset: the dataset associated with the MBO task.
        """
        self.task = task
        self.dataset = dataset
        self.oracle = self._task_oracle[self.task]

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the value of the oracle on the input design x.
        Input:
            x: the input batch of designs.
        Returns:
            The values of the oracle for each input design.
        """
        if self.task == MOLECULE_TASK:
            x = x[np.newaxis, ...] if x.ndim == 1 else x
            return np.array([
                [self.oracle(sf.decoder(self.dataset.data.decode(tok)))]
                for tok in x
            ])
        elif self.task == BRANIN_TASK:
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

    @property
    def _task_oracle(self) -> Dict[str, Callable]:
        """
        Returns the oracle function for each of the defined MBO tasks.
        Input:
            None.
        Returns:
            A dictionary mapping the task name to the oracle function.
        """
        return {
            "Branin-Branin-v0": Branin(negate=True),
            "MNISTIntensity-L2-v0": (
                lambda x: np.mean(np.square(x), axis=-1).astype(x.dtype)
            ),
            "PenalizedLogP-Guacamol-v0": MoleculeObjective("logP")
        }


def register(task: str) -> None:
    """
    Register a specification for a model-based optimization task.
    Input:
        task: the name of the model-based optimization task.
        verbose: whether to print verbose outputs. Default True.
    Returns:
        None.
    """
    dataset_kwargs = {
        "max_samples": None,
        "distribution": None,
        "max_percentile": 100,
        "min_percentile": 0
    }
    if task == "MNISTIntensity-L2-v0":
        dataset_kwargs["root"] = "./mnist/data"
    elif task == "PenalizedLogP-Guacamol-v0":
        dataset_kwargs["fname"] = "./molecules/data/train_selfie.gz"
        dataset_kwargs["fname"] = "./molecules/data/val_selfie.gz"

    design_bench.register(
        task,
        TASK_DATASETS[task],
        oracle=(lambda dataset: OracleWrapper(task, dataset)),
        dataset_kwargs=dataset_kwargs
    )
    logging.info(f"Registered task {task}.")
    return


for task in TASK_DATASETS.keys():
    register(task)
