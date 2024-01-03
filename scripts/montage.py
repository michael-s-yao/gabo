"""
Image montage generator utility script.

Author(s):
    Michael Yao

Licensed under the MIT License. Copyright University of Pennsylvania 2023.
"""
import argparse
import matplotlib.pyplot as plt
import torch
from pathlib import Path
from typing import Optional, Union

from data.mnist import MNISTDataModule
from models.generator import GeneratorModule
from utils import seed_everything, plot_config


def build_args() -> argparse.Namespace:
    """
    Defines arguments for montage generation.
    Input:
        None.
    Returns:
        An `argparse.Namespace` object with required argument values.
    """
    parser = argparse.ArgumentParser(
        description="Builds a montage of generated images."
    )

    parser.add_argument(
        "--height",
        type=int,
        default=10,
        help="Height of the montage in units of number of images. Default 10."
    )
    parser.add_argument(
        "--width",
        type=int,
        default=10,
        help="Width of the montage in units of number of images. Default 10."
    )
    ckpt_help = "Generative model path. If not provided, a montage of true "
    ckpt_help += "MNIST images is generated."
    parser.add_argument(
        "--ckpt", type=str, default=None, help=ckpt_help
    )
    parser.add_argument(
        "--savepath",
        type=str,
        default=None,
        help="File path to save montage to. Default not saved."
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Optional title for the montage. Default no title."
    )
    parser.add_argument(
        "--seed", default=42, type=int, help="Random seed. Default 42."
    )
    parser.add_argument(
        "--with_colorbar",
        action="store_true",
        help="Whether to include a colorbar with the montage."
    )

    return parser.parse_args()


def create_montage(
    H: int = 10,
    W: int = 10,
    seed: int = 42,
    model: Optional[Union[Path, str]] = None,
    savepath: Optional[Union[Path, str]] = "./montage.png",
    title: Optional[str] = None,
    with_colorbar: bool = False
) -> None:
    """
    Creates a montage of HxW generated images.
    Input:
        H: height of the montage in units of number of images. Default 10.
        W: width of the montage in units of number of images. Default 10.
        seed: random seed. Default 42.
        model: generative model path. If not provided, a montage of true MNIST
            images is generated.
        savepath: file path to save montage to. Default `./montage.png`.
        title: optional title for the montage. Default no title.
        with_colorbar: whether to include a colorbar with the montage.
    Returns:
        None.
    """
    n = H * W
    device = torch.device("cpu")

    if model:
        try:
            model = GeneratorModule.load_from_checkpoint(model)
        except RuntimeError:
            model = GeneratorModule.load_from_checkpoint(
                model, map_location=device
            )

        z = torch.randn((n, model.hparams.z_dim)).to(device)
        X = model(z)
    else:
        datamodule = MNISTDataModule(batch_size=n, num_workers=0, seed=seed)
        datamodule.prepare_data()
        datamodule.setup(stage="test")
        X, _ = next(iter(datamodule.test_dataloader()))

    X = torch.squeeze(X, dim=1)
    _, imgH, imgW = X.size()
    montage = torch.empty((imgH * H, imgW * W), device=device)
    for idx, img in enumerate(X):
        row, col = idx // W, idx % W
        rowS, rowE = row * imgH, (row + 1) * imgH
        colS, colE = col * imgW, (col + 1) * imgW
        montage[rowS:rowE, colS:colE] = img

    plt.plot()
    plt.imshow(montage.detach().numpy(), cmap="gray")
    plt.axis("off")
    if with_colorbar:
        plt.colorbar()
    if title:
        plt.title(title)

    if savepath is None:
        plt.show()
    else:
        plt.savefig(savepath, dpi=600, transparent=True, bbox_inches="tight")

    return


if __name__ == "__main__":
    args = build_args()
    seed_everything(seed=args.seed)
    plot_config()
    create_montage(
        H=args.height,
        W=args.width,
        seed=args.seed,
        model=args.ckpt,
        savepath=args.savepath,
        title=args.title,
        with_colorbar=args.with_colorbar
    )
