#!/usr/bin/env python3
"""
Main driver program for the COM baseline MBO method.

Author(s):
    Michael Yao @michael-s-yao

Licensed under the MIT License. Copyright University of Pennsylvania 2023.
"""
import sys
from typing import Optional

sys.path.append(".")
import mbo  # noqa
from design_baselines.coms_cleaned import coms_cleaned
from utils import seed_everything


def parse_seed() -> Optional[int]:
    """
    Finds, returns, and removes the random seed (if specified) from the
    command line arguments.
    Input:
        None.
    Returns:
        The random seed value (if specified).
    """
    for i in range(len(sys.argv) - 1):
        if sys.argv[i] == "--seed":
            seed = int(sys.argv[i + 1])
            sys.argv = sys.argv[:i] + sys.argv[(i + 2):]
            return int(seed)
    return None


def main():
    seed_everything(seed=parse_seed())
    coms_cleaned()


if __name__ == "__main__":
    main()
