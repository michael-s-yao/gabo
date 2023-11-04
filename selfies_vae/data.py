"""
Pytorch Lightning Data Module for Molecule Generation.

Author(s):
    Yimeng Zeng @yimeng-zeng
    Michael Yao @michael-s-yao

Citations(s):
    [1] Krenn M, Hase F, Nigam AK, Friederich P, Aspuru-Guzik A. Self-
        referencing embedded strings (SELFIES): A 100% robust molecular string
        representation. Machine Learning: Science and Technology 1(4): 045024.
        (2020). https://doi.org/10.1088/2632-2153/aba947

Licensed under the MIT License. Copyright University of Pennsylvania 2023.
"""
import gzip
import os
import json
import torch
import torch.nn.functional as F
import selfies as sf
import pytorch_lightning as pl
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from typing import Optional, Sequence, Union


class SELFIESDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int = 256,
        num_workers: int = os.cpu_count() // 2,
        train_datapath: Union[Path, str] = "data/train_selfie.gz",
        val_datapath: Union[Path, str] = "data/val_selfie.gz",
        test_datapath: Union[Path, str] = "data/test_selfie.gz",
        load_train_data: bool = False
    ):
        """
        Args:
            batch_size: batch size.
            num_workers: number of workers.
            train_datapath: filename of the training dataset.
            val_datapath: filename of the validation dataset.
            test_datapath: filename of the test dataset.
            load_train_data: whether to load the training data.
        """
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.home = os.path.dirname(__file__)
        self.train = SELFIESDataset(
            os.path.join(self.home, train_datapath), load_data=load_train_data
        )
        self.val = SELFIESDataset(
            os.path.join(self.home, val_datapath), load_data=True
        )
        self.test = SELFIESDataset(
            os.path.join(self.home, test_datapath), load_data=True
        )
        self.val.vocab, self.test.vocab = self.train.vocab, self.train.vocab
        self.val.vocab2idx = self.train.vocab2idx
        self.test.vocab2idx = self.train.vocab2idx
        # Drop data from val and test that we have no tokens for.
        self.val.data = [
            smile
            for smile in self.val.data
            if False not in [tok in self.train.vocab for tok in smile]
        ]
        self.test.data = [
            smile
            for smile in self.test.data
            if False not in [tok in self.train.vocab for tok in smile]
        ]

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
            batch_size=self.batch_size,
            pin_memory=True,
            shuffle=True,
            collate_fn=self.train.collate_fn,
            num_workers=self.num_workers
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
            self.val,
            batch_size=self.batch_size,
            pin_memory=True,
            shuffle=False,
            collate_fn=self.val.collate_fn,
            num_workers=self.num_workers
        )

    def test_dataloader(self) -> DataLoader:
        """
        Returns the test dataloader.
        Input:
            None.
        Returns:
            The test dataloader.
        """
        return DataLoader(
            self.test,
            batch_size=self.batch_size,
            pin_memory=True,
            shuffle=False,
            collate_fn=self.val.collate_fn,
            num_workers=self.num_workers
        )


class SELFIESDataset(Dataset):
    def __init__(
        self,
        fname: Optional[Union[Path, str]] = None,
        load_data: bool = False,
        default_selfies_vocab: Union[Path, str] = "vocab.json"
    ):
        """
        Args:
            fname: file path to the directory of SELFIES molecules.
            load_data: whether or not to load the data in the directory.
            default_selfies_vocab: file path to default SELFIES vocab.
        """
        self.data = []
        self.start, self.stop = "<start>", "<stop>"
        self.home = os.path.dirname(__file__)
        if load_data:
            if fname is None:
                raise ValueError(
                    "fname argument must be specified if loading data."
                )
            with gzip.open(fname, "r") as f:
                selfie_strings = [x.decode().strip() for x in f.readlines()]
            for string in selfie_strings:
                self.data.append(list(sf.split_selfies(string)))
            self.vocab = set([
                token for selfie in self.data for token in selfie
            ])
            self.vocab.discard(".")
            self.vocab = [
                self.start, self.stop, *sorted(list(self.vocab))
            ]
        else:
            with open(
                os.path.join(self.home, default_selfies_vocab), "r"
            ) as f:
                self.vocab = json.load(f)
        self.vocab2idx = {char: idx for idx, char in enumerate(self.vocab)}

    def tokenize_selfies(
        self, selfies_list: Sequence[str]
    ) -> Sequence[Sequence[str]]:
        """
        Splits SELFIES strings into their list representations of tokens.
        Input:
            selfies_list: a sequence of SELFIES strings to tokenize.
        Returns:
            A list of the tokenized representations of the SELFIES strings.
        """
        return [list(sf.split_selfies(string)) for string in selfies_list]

    def encode(self, selfies: Sequence[str]) -> torch.Tensor:
        """
        Encodes a tokenized SELFIES representation as a tensor.
        Input:
            selfies: a tokenized representation of a SELFIES string.
        Returns:
            The encoding of the input SELFIES string as a tensor.
        """
        return torch.tensor([self.vocab2idx[s] for s in [*selfies, self.stop]])

    def decode(self, tokens: torch.Tensor) -> Sequence[str]:
        """
        Decodes an encoded SELFIES string back into a SELFIES string.
        Input:
            tokens: a single tensor representation of a SELFIES string.
        Returns:
            The decoded SELFIES string.
        """
        dec = [self.vocab[t] for t in tokens]
        # Ignore the start token and everything past (and including) the first
        # stop token.
        stop = dec.index(self.stop) if self.stop in dec else None
        selfie = dec[0:stop]
        # Start at the last start token (I've seen one case where it started
        # with 2 start tokens).
        while self.start in selfie:
            selfie = selfie[(1 + dec.index(self.start)):]
        selfie = "".join(selfie)
        return selfie

    def __len__(self) -> int:
        """
        Returns the number of molecules in the dataset.
        Input:
            None.
        Returns:
            The number of molecules in the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Retrieves the specified molecule from the dataset as an encoded tensor.
        Input:
            idx: the index of the molecule to retrieve from the dataset.
        Returns:
            The specified molecule from the dataset as an encoded tensor.
        """
        return self.encode(self.data[idx])

    @property
    def vocab_size(self) -> int:
        """
        Returns the size of the vocabulary for the dataset.
        Input:
            None.
        Returns:
            The size of the vocabulary for the dataset.
        """
        return len(self.vocab)

    def collate_fn(self, data: Sequence[torch.Tensor]) -> torch.Tensor:
        """
        Custom collate function.
        Input:
            data: an input batch of tokenized representations of molecules.
        Returns:
            The collated batch of data.
        """
        max_size = max([x.shape[-1] for x in data])
        # Pad shorter molecules with the stop token.
        return torch.vstack([
            F.pad(
                x, (0, max_size - x.shape[-1]), value=self.vocab2idx[self.stop]
            )
            for x in data
        ])
