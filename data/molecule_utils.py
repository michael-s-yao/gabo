"""
Utility functions to convert between different molecule representations.

Author(s):
    Michael Yao

Licensed under the MIT License. Copyright University of Pennsylvania 2023.
"""
import torch
import torch.nn.functional as F
from typing import Dict, Optional, Sequence


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
