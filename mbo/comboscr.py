"""
Implementation of conservative offline model-based optimization over latent
spaces via source critic regularization (COMBO-SCR). Our method estimates the
Lagrange multiplier through solving the dual problem of the primal optimization
task.

Author(s):
    Michael Yao @michael-s-yao

Licensed under the MIT License. Copyright University of Pennsylvania 2023.
"""
import json
import logging
import math
import numpy as np
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from botorch.utils.transforms import unnormalize
from pathlib import Path
from typing import Union

sys.path.append(".")
import mbo  # noqa
import design_bench
from mbo.args import build_args
from data.data import DesignBenchDataModule
from models.policy import COMBOSCRPolicy
from models.joint import JointVAESurrogate
from models.oracle.branin import BraninOracle
from utils import get_device, seed_everything


def main():
    args = build_args()
    device = get_device(args.device)
    seed_everything(args.seed)
    task = design_bench.make(args.task)

    with open("./mbo/hparams.json", "rb") as f:
        hparams = json.load(f)[args.task]
    z_bound = hparams.pop("z_bound")
    batch_size = hparams.pop("batch_size")
    budget = hparams.pop("budget")
    num_generator_per_critic = hparams.pop("num_generator_per_critic")

    vae, surrogate = load_vae_and_surrogate_models(
        task, args.task, device=device, **hparams
    )

    if isinstance(z_bound, float):
        bounds = torch.tensor(
            [[-z_bound] * vae.latent_size, [z_bound] * vae.latent_size],
            device=device
        )
    else:
        bounds = torch.tensor(z_bound, device=device)

    policy = COMBOSCRPolicy(
        args.task,
        vae.latent_size,
        bounds,
        torch.from_numpy(task.x).to(device),
        surrogate,
        batch_size=batch_size,
        alpha=args.alpha,
        device=device,
        **hparams
    )

    oracle = BraninOracle(negate=True)

    # Choose the initial set of observations.
    a, Wds = [], []
    sobol = torch.quasirandom.SobolEngine(
        dimension=vae.latent_size, scramble=True, seed=args.seed
    )
    z = unnormalize(
        sobol.draw(n=batch_size).to(device), bounds=bounds
    )
    z_ref, _, _ = vae.encode(
        policy.reference_sample(8 * batch_size).flatten(start_dim=1)
    )
    policy.fit_critic(z.detach(), z_ref.detach())
    alpha = policy.alpha(z_ref)
    a.append(alpha)
    Wds.append(torch.mean(policy.wasserstein(z_ref, z)).item())
    y = (1.0 - alpha) * surrogate(z) - alpha * F.relu(
        policy.wasserstein(z_ref.detach(), z.detach()).unsqueeze(dim=-1)
    )
    y_gt = torch.unsqueeze(oracle(z), dim=-1)
    y, y_gt = y.unsqueeze(dim=1), y_gt.unsqueeze(dim=1)

    for step in range(math.ceil(budget / batch_size) - 1):
        policy.fit(z.detach(), y.flatten().unsqueeze(dim=-1).detach())

        # Optimize and get new observations.
        new_z = policy(y.flatten().unsqueeze(dim=-1))
        z_ref, _, _ = vae.encode(
            policy.reference_sample(8 * batch_size).flatten(start_dim=1)
        )

        alpha = policy.alpha(z_ref)
        a.append(alpha)
        Wds.append(torch.mean(policy.wasserstein(z_ref, new_z)).item())
        new_y = (1.0 - alpha) * surrogate(new_z) - alpha * F.relu(
            policy.wasserstein(z_ref.detach(), new_z.detach()).unsqueeze(
                dim=-1
            )
        )
        new_gt = torch.unsqueeze(oracle(new_z), dim=-1)
        new_y, new_gt = new_y.unsqueeze(dim=1), new_gt.unsqueeze(dim=1)

        # Update training points.
        z = torch.cat((z, new_z), dim=0)
        y = torch.cat((y, new_y), dim=1)
        y_gt = torch.cat((y_gt, new_gt), dim=1)

        # Update progress.
        policy.save_current_state_dict()

        # Train the source critic.
        if step % num_generator_per_critic == 0:
            policy.fit_critic(new_z.detach(), z_ref.detach())

    # Save optimization results.
    if args.logging_dir is not None:
        os.makedirs(args.logging_dir, exist_ok=True)
        z = z.reshape(y.size(dim=0), y.size(dim=1), z.size(dim=-1))
        np.save(
            os.path.join(args.logging_dir, "solution.npy"),
            z.detach().cpu().numpy()
        )
        np.save(
            os.path.join(args.logging_dir, "predictions.npy"),
            y.detach().cpu().numpy()
        )
        np.save(
            os.path.join(args.logging_dir, "scores.npy"),
            y_gt.detach().cpu().numpy()
        )
        logging.info(f"Saved experiments results to {args.logging_dir}")


def load_vae_and_surrogate_models(
    task: design_bench.task.Task,
    task_name: str,
    ckpt_dir: Union[Path, str] = "./checkpoints",
    device: torch.device = torch.device("cpu"),
    **kwargs
) -> Union[nn.Module]:
    """
    Loads a trained VAE model and/or trained surrogate models for model-based
    optimization (MBO).
    Input:
        task: design-bench MBO task.
        task_name: name of the design-bench MBO task.
        ckpt_dir: directory to saved checkpoints.
        device: device. Default CPU.
    Returns:
        vae: a trained VAE model for encoding and decoding designs into and
            from a continuous latent space. For tasks with no VAE model
            defined, the identity function is returned instead.
        surrogate: a trained surrogate model.
    """
    ckpt = os.path.join(ckpt_dir, f"{task_name}.ckpt")
    if not os.path.isfile(ckpt):
        logging.info(f"Model {ckpt} does not exist, training now...")
        fit_vae_and_surrogate_models(
            task, task_name, ckpt_dir=ckpt_dir, device=str(device), **kwargs
        )
        logging.info(f"Trained model can be found at {ckpt}")
    model = JointVAESurrogate.load_from_checkpoint(
        ckpt, map_location=device, task=task, task_name=task_name
    )
    model = model.to(device).eval()
    return model.vae, model.surrogate


def fit_vae_and_surrogate_models(
    task: design_bench.task.Task,
    task_name: str,
    ckpt_dir: Union[Path, str] = "./checkpoints",
    device: str = "auto",
    lr: float = 0.001,
    num_epochs: int = 100,
    **kwargs
) -> None:
    """
    Jointly trains a VAE model and surrogate models for model-based
    optimization (MBO).
    Input:
        task: design-bench MBO task.
        task_name: name of the design-bench MBO task.
        ckpt_dir: directory to saved checkpoints.
        device: device. Default CPU.
        lr: learning rate. Default 0.001.
        num_epochs: number of training epochs. Default 100.
    Returns:
        None.
    """
    dm = DesignBenchDataModule(task=task, device=device)
    model = JointVAESurrogate(task=task, task_name=task_name, lr=lr, **kwargs)
    devices = "".join(filter(str.isdigit, device))
    devices = [int(devices)] if len(devices) > 0 else "auto"
    accelerator = device.split(":")[0].lower()
    accelerator = "gpu" if accelerator == "cuda" else accelerator
    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        callbacks=[
            pl.callbacks.ModelCheckpoint(
                monitor="val/objective_mse",
                dirpath=ckpt_dir,
                filename=task_name
            )
        ],
        max_epochs=num_epochs,
        logger=False
    )
    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    main()
