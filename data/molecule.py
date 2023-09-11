"""
SELFIES Molecules Pytorch Lightning Data Module.

Author(s):
    Michael Yao @michael-s-yao
    Yimeng Zeng @yimeng-zeng

Adapted from Haydn Jones @haydn-jones molformers repo.

Licensed under the MIT License. Copyright University of Pennsylvania 2023.
"""
import os
import gzip
from pathlib import Path
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
        max_molecule_length: int = 128,
        seed: int = 42,
    ):
        """
        Args:
            vocab: vocabulary dictionary.
            root: directory path.
            batch_size: batch size. Default 128.
            num_workers: number of workers. Default half the CPU count.
            max_molecule_length: maximum molecule length. Default 128.
            seed: random seed. Default 42.
        """
        super().__init__()
        self.vocab = vocab
        self.root = root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed

        self.max_molecule_length = max_molecule_length

        self.train = SELFIESDataset(
            os.path.join(self.root, "train_selfie.gz"),
            self.vocab,
            self.max_molecule_length,
            name="train"
        )
        self.val = SELFIESDataset(
            os.path.join(self.root, "val_selfie.gz"),
            self.vocab,
            self.max_molecule_length,
            name="val"
        )
        self.test = SELFIESDataset(
            os.path.join(self.root, "test_selfie.gz"),
            self.vocab,
            self.max_molecule_length,
            name="test"
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


class SELFIESDataset(Dataset):
    def __init__(
        self,
        datapath: Union[Path, str],
        vocab: Dict[str, int],
        max_molecule_length: int,
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
        mol = f"{self.start}{self.data[idx]}{self.stop}"
        item = torch.tensor([
            self.vocab[f"[{tok.split(']')[0]}]"] for tok in mol[1:].split("[")
        ])
        if len(item) > self.max_molecule_length:
            raise ValueError(
                f"Molecule exceeds maximum length {self.max_molecule_length}."
            )
        padding = torch.ones(self.max_molecule_length - len(item))
        return torch.cat((item, self.vocab[self.stop] * padding))

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
        # Perform one-hot encoding.
        enc = torch.zeros(
            (len(batch), self.max_molecule_length, len(self.vocab.keys()))
        )
        for batch_idx in range(len(batch)):
            for seq_idx in range(self.max_molecule_length):
                enc[batch_idx, seq_idx, int(batch[batch_idx][seq_idx])] = 1.0
        return enc
