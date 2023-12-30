"""
Main driver program for baseline MBO methods.

Author(s):
    Michael Yao @michael-s-yao

Licensed under the MIT License. Copyright University of Pennsylvania 2023.
"""
import sys

sys.path.append(".")
import mbo # noqa
from design_baselines.coms_cleaned import coms_cleaned


def main():
    coms_cleaned()


if __name__ == "__main__":
    main()
