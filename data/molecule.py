"""
SELFIES Molecules Pytorch Lightning Data Module.

Author(s):
    Michael Yao @michael-s-yao
    Yimeng Zeng @yimeng-zeng

Adapted from Haydn Jones @haydn-jones molformers repo.

Licensed under the MIT License. Copyright University of Pennsylvania 2023.
"""
from itertools import chain
import os
import gzip
from pathlib import Path
import selfies as sf
import torch
import torch.nn.functional as F
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
        max_molecule_length: int = 109,
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
                which is 109.
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
        return torch.tensor([self.vocab[tok] for tok in molecule])

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


def tokens_to_selfies(
    tokens: torch.Tensor, vocab: Dict[str, int], pad: Optional[str] = "[pad]"
) -> Sequence[str]:
    """
    Converts a batch of token representations of molecules into SELFIES string
    representations.
    Input:
        tokens: molecules tensor with dimensions BN, where B is the batch size
            and N is the length of the molecule representation.
        vocab: vocab dictionary.
        pad: optional padding token. Default `[pad]`.
    Returns:
        A sequence of B string representations of the B molecules.
    """
    inv_vocab = {val: key for key, val in vocab.items()}
    selfies = []
    for mol in tokens:
        rep = "".join([inv_vocab[int(tok)] for tok in mol])
        if pad:
            rep = rep.replace(pad, "")
        selfies.append(rep)
    return selfies


def one_hot_encodings_to_tokens(encodings: torch.Tensor) -> torch.Tensor:
    """
    Converts a batch of one-hot encoding representations of molecules into
    token representations.
    Input:
        encodings: one-hot encodings tensor with dimensions BNK, where B is the
            batch size, N is the length of the molecule representation, and K
            is the size of the vocabulary dictionary.
    Returns:
        A tensor with dimensions BN with the token representations of the B
        molecules.
    """
    return torch.argmax(encodings, dim=-1)


def logits_to_tokens(logits: torch.Tensor) -> torch.Tensor:
    """
    Converts a batch of logits for molecule representations into token
    representations.
    Input:
        logits: logits tensor with dimensions BNK, where B is the
            batch size, N is the length of the molecule representation, and K
            is the size of the vocabulary dictionary.
    Returns:
        A tensor with dimensions BN with the token representations of the B
        molecules.
    """
    return torch.argmax(F.softmax(logits, dim=-1), dim=-1)


def one_hot_encodings_to_selfies(
    encodings: torch.Tensor,
    vocab: Dict[str, int],
    pad: Optional[str] = "[pad]"
) -> Sequence[str]:
    """
    Converts a batch of one-hot encoding representations of molecules into
    SELFIES string representations.
    Input:
        encodings: one-hot encodings tensor with dimensions BNK, where B is the
            batch size, N is the length of the molecule representation, and K
            is the size of the vocabulary dictionary.
        vocab: vocab dictionary.
        pad: optional padding token. Default `[pad]`.
    Returns:
        A sequence of B string representations of the B molecules.
    """
    return tokens_to_selfies(
        one_hot_encodings_to_tokens(encodings), vocab, pad
    )


def logits_to_selfies(
    logits: torch.Tensor, vocab: Dict[str, int], pad: Optional[str] = "[pad]"
) -> torch.Tensor:
    """
    Converts a batch of logits for molecule representations into SELFIES string
    representations.
    Input:
        logits: logits tensor with dimensions BNK, where B is the
            batch size, N is the length of the molecule representation, and K
            is the size of the vocabulary dictionary.
        vocab: vocab dictionary.
        pad: optional padding token. Default `[pad]`.
    Returns:
        A sequence of B string representations of the B molecules.
    """
    return tokens_to_selfies(logits_to_tokens(logits), vocab, pad)
