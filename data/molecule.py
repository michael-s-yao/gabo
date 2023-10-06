"""
SELFIES Molecules Pytorch Lightning Data Module.

Author(s):
    Michael Yao @michael-s-yao
    Yimeng Zeng @yimeng-zeng

Citations(s):
    [1] Krenn M, Hase F, Nigam AK, Friederich P, Aspuru-Guzik A. Self-
        referencing embedded strings (SELFIES): A 100% robust molecular string
        representation. Machine Learning: Science and Technology 1(4): 045024.
        (2020). https://doi.org/10.1088/2632-2153/aba947
    [2] Ramakrishnan R, Dral PO, Rupp M, Anatole von Lilienfeld O. Quantum
        chemistry structures and properties of 134 kilo molecules. Nat Sci
        Data 1: 140022. (2014). https://doi.org/10.1038/sdata.2014.22

Adapted from Haydn Jones @haydn-jones molformers repo.

Licensed under the MIT License. Copyright University of Pennsylvania 2023.
"""
from itertools import chain
from math import isclose
import numpy as np
import os
import gzip
from pathlib import Path
import selfies as sf
import torch
from torch.utils.data import DataLoader, Dataset
import lightning.pytorch as pl
from typing import Dict, Optional, Sequence, Union


class SELFIESDataModule(pl.LightningDataModule):
    def __init__(
        self,
        vocab: Dict[str, int],
        root: Union[Path, str] = "./MolOOD/data",
        batch_size: int = 128,
        num_workers: int = os.cpu_count() // 2,
        max_molecule_length: int = 111,
        seed: int = 42,
    ):
        """
        Args:
            vocab: vocabulary dictionary.
            root: directory path.
            batch_size: batch size. Default 128.
            num_workers: number of workers. Default half the CPU count.
            max_molecule_length: maximum molecule length. Default maximum
                molecule length in training, validation, and test datasets,
                which is 111.
            seed: random seed. Default 42.
        """
        super().__init__()
        self.vocab = vocab
        self.root = root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed

        self.train = SELFIESDataset(
            os.path.join(self.root, "train_selfie.gz"), self.vocab
        )
        self.val = SELFIESDataset(
            os.path.join(self.root, "val_selfie.gz"), self.vocab
        )
        self.test = SELFIESDataset(
            os.path.join(self.root, "test_selfie.gz"), self.vocab
        )

        self.max_molecule_length = max_molecule_length
        if self.max_molecule_length < 1:
            for molecule in chain(self.train, self.val, self.test):
                self.max_molecule_length = max(
                    torch.numel(molecule), self.max_molecule_length
                )
        self.train.max_molecule_length = self.max_molecule_length
        self.val.max_molecule_length = self.max_molecule_length
        self.test.max_molecule_length = self.max_molecule_length

    def train_dataloader(self) -> DataLoader:
        """
        Retrieves the training dataloader.
        Input:
            None.
        Returns:
            The training dataloader.
        """
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.train.collate_fn
        )

    def val_dataloader(self) -> DataLoader:
        """
        Retrieves the validation dataloader.
        Input:
            None.
        Returns:
            The validation dataloader.
        """
        return DataLoader(
            self.val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.val.collate_fn
        )

    def test_dataloader(self) -> DataLoader:
        """
        Retrieves the test dataloader.
        Input:
            None.
        Returns:
            The test dataloader.
        """
        return DataLoader(
            self.test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.test.collate_fn
        )


class QM9DataModule(pl.LightningDataModule):
    def __init__(
        self,
        fn: Union[Path, str] = "./data/molecules/2RGSMILES_QM9.txt",
        partition: Sequence[float] = [0.8, 0.1, 0.1],
        batch_size: int = 128,
        num_workers: int = os.cpu_count() // 2,
        seed: int = 42,
    ):
        """
        Args:
            fn: file of the SELFIES representation of the QM9 dataset.
            partition: partition for training, validation, and test datasets.
                Default 80% training, 10% validation, and 10% test.
            batch_size: batch size. Default 128.
            num_workers: number of workers. Default half the CPU count.
            seed: random seed. Default 42.
        """
        super().__init__()
        self.fn = fn
        self.partition = partition
        if not isclose(sum(partition), 1.0):
            raise ValueError(
                f"Parition fractions must sum to 1.0, got {partition}."
            )
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.rng = np.random.RandomState(self.seed)

    def prepare_data(self) -> None:
        """
        Loads the dataset and other relevant parameters from the dataset.
        Input:
            None.
        Returns:
            None.
        """
        with open(self.fn, "rb") as f:
            self.molecules = [
                str(x.strip()).split(",")[1][:-1] for x in f.readlines()[1:]
            ]
            self.molecules = np.array(list(set(self.molecules)))
            self.rng.shuffle(self.molecules)
        vocab = sorted(list(set(list(" ".join(self.molecules)))))
        self.vocab = {tok: idx for idx, tok in enumerate(vocab)}
        self.max_molecule_length = max(map(lambda x: len(x), self.molecules))

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Partitions the dataset into training, validation, and/or test sets.
        Input:
            stage: model training or evaluation stage. One of [`fit`, `test`,
                None].
        Returns:
            None.
        """
        train_frac, _, test_frac = self.partition
        num_train = round(train_frac * len(self.molecules))
        num_test = round(test_frac * len(self.molecules))
        if stage is None or stage == "fit":
            self.train = QM9Dataset(
                molecules=self.molecules[:num_train],
                vocab=self.vocab,
                max_molecule_length=self.max_molecule_length
            )
            self.val = QM9Dataset(
                molecules=self.molecules[num_train:-num_test],
                vocab=self.vocab,
                max_molecule_length=self.max_molecule_length
            )
        if stage is None or stage == "test":
            self.test = QM9Dataset(
                molecules=self.molecules[-num_test:],
                vocab=self.vocab,
                max_molecule_length=self.max_molecule_length
            )

    def train_dataloader(self) -> DataLoader:
        """
        Retrieves the training dataloader.
        Input:
            None.
        Returns:
            The training dataloader.
        """
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )

    def val_dataloader(self) -> DataLoader:
        """
        Retrieves the validation dataloader.
        Input:
            None.
        Returns:
            The validation dataloader.
        """
        return DataLoader(
            self.val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )

    def test_dataloader(self) -> DataLoader:
        """
        Retrieves the test dataloader.
        Input:
            None.
        Returns:
            The test dataloader.
        """
        return DataLoader(
            self.test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )


class SELFIESDataset(Dataset):
    def __init__(
        self,
        datapath: Union[Path, str],
        vocab: Dict[str, int],
        max_molecule_length: int = -1,
        name: Optional[str] = None,
    ):
        """
        Args:
            datapath: file name containing the dataset.
            vocab: a dictionary mapping molecular components to numbers.
            max_molecule_length: maximum molecule length.
            name: optional name for the dataset object. Default None.
        """
        super().__init__()
        self.datapath = datapath
        self.vocab = vocab
        self.max_molecule_length = max_molecule_length
        self.name = name
        self.start, self.stop, self.pad = "[start]", "[stop]", "[pad]"
        with gzip.open(self.datapath, "rb") as f:
            self.data = [line.strip() for line in f.readlines()]

    def __len__(self) -> int:
        """
        Returns the length of the dataset.
        Input:
            None.
        Returns:
            Length of the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Retrieves a specified item from the dataset.
        Input:
            idx: the index of the element to retrieve.
        Returns:
            The `idx`th molecule from the dataset.
        """
        molecule = sf.split_selfies(self.data[idx].decode('ASCII'))
        molecule = [self.vocab[tok] for tok in molecule]
        return torch.tensor(
            [self.vocab[self.start]] + molecule + [self.vocab[self.stop]]
        )

    def collate_fn(self, batch: Sequence[torch.Tensor]) -> torch.Tensor:
        """
        Custom `collate_fn()` implementation with built-in one-hot encoding.
        Input:
            batch: A tuple of tensors to collate.
        Returns:
            Collated one-hot encoded batch with dimensions BNA, where B is the
            batch size, N is the maximum number of characters in the batch, and
            A is the size of the vocabulary dictionary.
        """
        # Perform one-hot encoding and padding.
        enc = torch.zeros(
            (len(batch), self.max_molecule_length, len(self.vocab.keys()))
        )
        for batch_idx in range(len(batch)):
            for seq_idx in range(len(batch[batch_idx])):
                enc[batch_idx, seq_idx, int(batch[batch_idx][seq_idx])] = 1.0
            for seq_idx in range(
                len(batch[batch_idx]), self.max_molecule_length
            ):
                enc[batch_idx, seq_idx, self.vocab[self.pad]] = 1.0
        return enc


class QM9Dataset(Dataset):
    def __init__(
        self,
        molecules: Sequence[str],
        vocab: Dict[str, int],
        max_molecule_length: int,
        padding_token: str = " "
    ):
        """
        Args:
            molecules: a list of molecules in SELFIES representation.
            vocab: a dictionary mapping molecular components to numbers.
            max_molecule_length: maximum molecule length.
            padding_token: padding token. Default ` `.
        """
        super().__init__()
        self.molecules = molecules
        self.vocab = vocab
        self.max_molecule_length = max_molecule_length
        self.pad = padding_token

    def __len__(self) -> int:
        """
        Returns the length of the dataset.
        Input:
            None.
        Returns:
            Length of the dataset.
        """
        return len(self.molecules)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Retrieves a specified item from the dataset.
        Input:
            idx: the index of the element to retrieve.
        Returns:
            The `idx`th molecule from the dataset as a padded one-hot encoded
            vector.
        """
        mol = self.molecules[idx]
        enc = torch.zeros(
            (self.max_molecule_length, len(self.vocab.keys())),
            dtype=torch.float
        )
        for idx, tok in enumerate(
            mol + self.pad * (self.max_molecule_length - len(mol))
        ):
            enc[idx, self.vocab[tok]] = 1.0
        return enc
