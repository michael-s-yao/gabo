"""
Simple utility function to map saved model weights to CPU storage.

Author(s):
    Michael Yao

Licensed under the MIT License. Copyright University of Pennsylvania 2023.
"""
import argparse
import torch


def convert() -> None:
    """
    Utility function to map saved model weights to CPU storage.
    Input:
        None.
    Returns:
        None.
    """
    parser = argparse.ArgumentParser(
        description="Tool to map saved model weights to CPU storage."
    )
    parser.add_argument("--model", type=str, required=True)
    args = parser.parse_args()
    model = torch.load(args.model, map_location=torch.device("cpu"))
    torch.save(model, args.model)
    return


if __name__ == "__main__":
    convert()
