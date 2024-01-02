"""
Defines interfacing classes for custom and design-bench benchmarking datasets.

Author(s):
    Michael Yao @michael-s-yao

Citation(s):
    [1] Trabucco B, Kumar A, Geng X, Levine S. Design-Bench: Benchmarks for
        data-driven offline model-based optimization. Proc ICML 162: 21658-76.
        (2022). https://proceedings.mlr.press/v162/trabucco22a.html

Licensed under the MIT License. Copyright University of Pennsylvania 2023.
"""
import numpy as np
import os
import pickle
import selfies as sf
import sys
import torch
import torchvision as thv
import pytorch_lightning as pl
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Tuple, Union
from botorch.test_functions.synthetic import Branin
from design_bench.datasets.discrete_dataset import DiscreteDataset
from design_bench.datasets.continuous_dataset import ContinuousDataset
from design_bench.task import Task

sys.path.append(".")
from molecules.data import SELFIESDataset
from molecules.utils import MoleculeObjective


class DesignBenchDataModule(pl.LightningDataModule):
    def __init__(
        self,
        task: Task,
        batch_size: int = 64,
        num_workers: int = 0,
        device: str = "cpu",
    ):
        """
        Args:
            task: design-bench task.
            batch_size: batch size. Default 64.
            num_workers: number of workers. Default 0.
            device: device. Default CPU.
        """
        super().__init__()
        self.task = task
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = device

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Partitions the dataset into training and validation partitions.
        Input:
            stage: optional stage specification of model training or eval.
        Returns:
            None.
        """
        self.train, self.val = self.task.dataset.split(val_fraction=0.1)
        self.train = DesignBenchDataset(self.train)
        self.val = DesignBenchDataset(self.val)
        return

    def train_dataloader(self) -> DataLoader:
        """
        Returns the training dataloader.
        Input:
            None.
        Returns:
            The training dataloader.
        """
        return DataLoader(
            self.train,
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            generator=torch.Generator(device=self.device)
        )

    def val_dataloader(self) -> DataLoader:
        """
        Returns the validation dataloader.
        Input:
            None.
        Returns:
            The validation dataloader.
        """
        return DataLoader(
            self.val, batch_size=self.batch_size, num_workers=self.num_workers
        )


class DesignBenchDataset(Dataset):
    def __init__(self, data: Union[DiscreteDataset, ContinuousDataset]):
        """
        Args:
            data: dataset containing design values x and associated objective
                values y.
        """
        self.data = data
        self.x = torch.from_numpy(data.x)
        self.y = torch.from_numpy(data.y)

    def __len__(self) -> int:
        """
        Returns the number of elements in the dataset.
        Input:
            None.
        Returns:
            The number of elements in the dataset.
        """
        return self.x.size(dim=0)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor]:
        """
        Returns a specified design value and objective value from the dataset.
        Input:
            idx: index of the datum to return.
        Returns:
            A tuple containing the specified design value and objective value.
        """
        return self.x[idx, :], self.y[idx, :]


class BraninDataset(ContinuousDataset):
    """
    Defines the negative 2D Branin objective function.

    Citation(s):
        [1] Branin FH and Hoo SK. A method for finding multiple extreme of a
            function of n variables. In: Numerical Method for Nonlinear
            Optimization, Cambridge. (1972).
    """
    def __init__(
        self,
        n: int = 1000,
        exclude_top_p: float = 0.2,
        seed: int = 42,
        **kwargs
    ):
        """
        Args:
            n: number of datums in the dataset. Default 1000.
            exclude_top_p: fraction of highest scoring datums to exclude from
                the dataset.
            seed: random seed. Default 42.
        """
        self.n, self.exclude_top_p, self.seed = n, exclude_top_p, seed
        self.rng = np.random.RandomState(self.seed)
        self.func = Branin(negate=True)
        self.sampler = torch.quasirandom.SobolEngine(
            self.func.bounds.size(dim=0), scramble=True, seed=self.seed
        )
        self.x1_range = self.func.bounds[:, 0].detach().cpu().numpy()
        self.x2_range = self.func.bounds[:, -1].detach().cpu().numpy()
        x = self.sampler.draw(self.n)
        x[:, 0] = self.x1_range[0] + (
            (self.x1_range[1] - self.x1_range[0]) * x[:, 0]
        )
        x[:, 1] = self.x2_range[0] + (
            (self.x2_range[1] - self.x2_range[0]) * x[:, 1]
        )
        x, y = x.detach().cpu().numpy(), self.func(x).detach().cpu().numpy()
        if self.exclude_top_p > 0.0:
            idxs = np.argsort(y)[:-round(self.exclude_top_p * self.n)]
            x, y = x[idxs], y[idxs]
        idxs = self.rng.permutation(len(x))
        x, y = x[idxs], y[idxs][..., np.newaxis]
        super(BraninDataset, self).__init__(x, y, **kwargs)


class MNISTIntensityDataset(ContinuousDataset):
    """
    Defines the MNIST image intensity dataset.

    Citation(s):
        [1] Deng L. The MNIST database of handwritten digit images for machine
            learning research. IEEE Sig Proc Mag 29(6): 141-2. (2012).
            https://doi.org/10.1109/MSP.2012.2211477
    """
    def __init__(self, root: Union[Path, str], **kwargs):
        """
        Args:
            root: root directory containing the MNIST dataset.
        """
        self.root = root
        self.data = thv.datasets.MNIST(
            root,
            train=True,
            download=True,
            transform=thv.transforms.ToTensor()
        )
        x = self.data.data.flatten(start_dim=(self.data.data.ndim - 2))
        x = x.detach().cpu().numpy() / torch.max(x).item()
        x = x.astype(np.float32)
        y = np.mean(np.square(x), axis=-1)[..., np.newaxis].astype(np.float32)
        super(MNISTIntensityDataset, self).__init__(x, y, **kwargs)


class PenalizedLogPDataset(DiscreteDataset):
    """
    Defines the penalized logP molecule dataset.

    Citations(s):
        [1] Krenn M, Hase F, Nigam AK, Friederich P, Aspuru-Guzik A. Self-
            referencing embedded strings (SELFIES): A 100% robust molecular
            string representation. Machine Learning: Science and Technology
            1(4): 045024. (2020). https://doi.org/10.1088/2632-2153/aba947
        [2] Brown N, Fiscato M, Segler MHS, Vaucher AC. GuacaMol: Benchmarking
            models for de novo molecular design. J Chem Inf Model 59(3): 1096-
            108. (2019). https://doi.org/10.1021/acs.jcim.8b00839
    """
    def __init__(
        self,
        fname: Union[Path, str],
        cache: Optional[Union[Path, str]] = "data/molecules/cache.pkl",
        **kwargs
    ):
        """
        Args:
            fname: file path to the directory of SELFIES molecules.
            cache: optional file path to a cache file of stored logP values.
        """
        self.fname, self.cache = fname, cache
        self.data = SELFIESDataset(self.fname, load_data=True)
        self.data.data = [
            smile
            for smile in self.data.data
            if False not in [tok in self.data.vocab for tok in smile]
        ]

        x = self.data.collate_fn([self.data[i] for i in range(len(self.data))])
        x = x.detach().cpu().numpy()

        smiles = [
            sf.decoder("".join(self.data.data[i]))
            for i in range(len(self.data))
        ]
        if self.cache is None or not os.path.isfile(self.cache):
            self.objective = MoleculeObjective("logP")
            y = np.array([[self.objective(smi)] for smi in smiles])
        else:
            with open(self.cache, "rb") as f:
                cache = pickle.load(f)
            y = np.array([[cache[smi]] for smi in smiles])
        y = y.astype(np.float32)

        super(PenalizedLogPDataset, self).__init__(
            x, y, num_classes=self.data.vocab_size, **kwargs
        )


class WarfarinDosingDataset(ContinuousDataset):
    pass
