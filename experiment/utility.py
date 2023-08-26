"""
Miscellaneous utility functions.

Author(s):
    Michael Yao

Licensed under the MIT License. Copyright University of Pennsylvania 2023.
"""
import matplotlib
import numpy as np
import os
import random
import pytorch_lightning as pl
import torch
import torch.distributed as dist
import warnings


def seed_everything(seed: int = 42, use_deterministic: bool = True) -> None:
    """
    Random state initialization function. Should be called during
    initialization.
    Input:
        seed: random seed. Default 42.
        use_deterministic: whether to only use deterministic algorithms.
            Default True.
    Returns:
        None.
    """
    pl.seed_everything(seed=seed, workers=True)
    try:
        torch.use_deterministic_algorithms(use_deterministic)
    except AttributeError:
        torch.set_deterministic(use_deterministic)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = use_deterministic

    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"


def plot_config(fontsize: int = 24, fontfamily: str = "STIXGeneral") -> None:
    """
    Plot configuration variables.
    Input:
        fontsize: plot font size. Default 24.
        fontfamily: plot font family. Default `STIXGeneral`.
    Returns:
        None.
    """
    matplotlib.rcParams["mathtext.fontset"] = "stix"
    matplotlib.rcParams["font.family"] = fontfamily
    matplotlib.rcParams.update({"font.size": fontsize})


def disable_warnings() -> None:
    """
    Disable selected warnings.
    Input:
        None.
    Returns:
        None.
    """
    warnings.filterwarnings("ignore", ".*does not have many workers.*")


def setup(
    rank: int,
    world_size: int,
    MASTER_ADDR: str = "localhost",
    MASTER_PORT: str = "12345"
) -> None:
    """
    Setup for data parallelism that runs across single or multiple machines.
    Input:
        MASTER_ADDR: a routable IP for all processes in the group to define
            the address of the rank 0 node. Default `localhost`.
        MASTER_PORT: a free port on machine with rank 0. Default `12355`.
    Returns:
        None.
    """
    os.environ["MASTER_ADDR"] = MASTER_ADDR
    os.environ["MASTER_PORT"] = MASTER_PORT

    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup() -> None:
    """
    Clean up for data parallelism.
    Input:
        None.
    Returns:
        None.
    """
    dist.destroy_process_group()
