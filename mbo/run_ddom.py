#!/usr/bin/env python3
"""
Main driver program for the DDOM baseline MBO method.

Author(s):
    Michael Yao @michael-s-yao

Citation(s):
    [1] Krishnamoorthy S, Mashkaria S, Grover A. Diffusion models for black-
        box optimization. Proc ICML 734:17842-857. (2023).
        https://dl.acm.org/doi/10.5555/3618408.3619142

Adapted from the ddom GitHub repo by @siddarthk97 at https://github.com/
siddarthk97/ddom/blob/main/design_baselines/diff/trainer.py

Licensed under the MIT License. Copyright University of Pennsylvania 2024.
"""
import argparse
import logging
import numpy as np
import os
import sys
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pathlib import Path
from typing import Callable, Optional, Sequence, Union

sys.path.append(".")
sys.path.append("ddom")
sys.path.append("ddom/design_baselines/diff")
import mbo
import design_bench
from helpers import seed_everything, get_device
from ddom.design_baselines.diff.trainer import RvSDataModule
from ddom.design_baselines.diff.nets import (
    DiffusionTest, DiffusionScore, Swish
)
from ddom.design_baselines.diff.lib.sdes import (
    PluginReverseSDE, VariancePreservingSDE
)
from ddom.design_baselines.diff.forward import ForwardModel


def build_args() -> argparse.Namespace:
    """
    Defines the experimental arguments for offline MBO baseline method
    evaluation.
    Input:
        None.
    Returns:
        A namespace containing the experimental argument values.
    """
    parser = argparse.ArgumentParser(description="DDOM Baseline Experiments")

    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=[task.task_name for task in design_bench.registry.all()],
        help="The name of the design-bench task for the experiment."
    )
    parser.add_argument(
        "--logging-dir",
        type=str,
        default=None,
        help="Logging directory to save optimization results to."
    )
    parser.add_argument(
        "--budget", type=int, default=256, help="Query budget. Default 256."
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=2048,
        help="Hidden size of the surrogate objective model. Default 2048."
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed. Default 42."
    )
    parser.add_argument(
        "--device", type=str, default="auto", help="Device. Default `auto`."
    )

    return parser.parse_args()


@torch.no_grad()
def heun_sampler(
    task: design_bench.task.Task,
    sde: Union[DiffusionTest, DiffusionScore],
    num_samples: int = 256,
    num_steps: int = 1000,
    gamma: float = 1.0,
    device: torch.device = torch.device("cpu"),
    grad_mask: Optional[torch.Tensor] = None
) -> Sequence[torch.Tensor]:
    """
    Input:
        task: design-bench MBO task.
        sde: stochastic differential equation (SDE) diffusion model.
        num_samples: number of samples. Default 256.
        num_steps: number of integration steps for sampling. Default 1000.
        gamma: drift parameter. Default 1.0.
        device: device. Default CPU.
        grad_mask: optional gradient mask for conditional optimization tasks.
    Returns:
        A sequence of the designs generated at each step.
    """
    grad_mask = grad_mask.to(device) if grad_mask is not None else None
    if task.is_discrete:
        X0 = torch.randn(
            num_samples, task.x.shape[-1] * task.x.shape[-2], device=device
        )
    else:
        X0 = torch.randn(num_samples, task.x.shape[-1], device=device)
    y = torch.full((num_samples,), task.y.max(), device=device)

    delta = sde.gen_sde.T.item() / num_steps
    Xs, ts = [], torch.linspace(0, 1, num_steps + 1) * sde.gen_sde.T.item()
    Xt = X0.detach().clone().to(device)
    t = torch.zeros(X0.size(dim=0), *([1] * (X0.ndim - 1)), device=device)
    tn = torch.zeros(X0.size(dim=0), *([1] * (X0.ndim - 1)), device=device)

    for i in range(num_steps):
        t.fill_(ts[i].item())
        if i < num_steps - 1:
            tn.fill_(ts[i + 1].item())
        mu = sde.gen_sde.mu(t, Xt, y, gamma=gamma)
        sigma = sde.gen_sde.sigma(t, Xt)
        dX = (delta * mu) + (delta ** 0.5) * sigma * torch.randn_like(Xt)
        dX = dX * grad_mask if grad_mask is not None else dX
        Xt = Xt + dX
        if i < num_steps - 1:
            sigma2 = sde.gen_sde.sigma(tn, Xt)
            dX = (sigma2 - sigma) / 2 * delta ** 0.5 * torch.randn_like(Xt)
            dX = dX * grad_mask if grad_mask is not None else dX
            Xt = Xt + dX
        Xs.append(Xt.detach().cpu())

    return Xs


class ConditionalPluginReverseSDE(PluginReverseSDE):
    """Implements a conditional SDE diffusion model."""

    def __init__(
        self,
        base_sde: VariancePreservingSDE,
        drift_a: nn.Module,
        T: torch.Tensor,
        grad_mask: torch.Tensor,
        vtype: str = "rademacher",
        debias: bool = False
    ):
        """
        Args:
            base_sde: a Stochastic Differential Equation (SDE) diffusion model.
            drift_a: the drift coefficient function modeled as a neural net.
            T: maximum time step.
            grad_mask: a mask tensor indicating with dimensions of the input
                are to be optimized over.
            vtype: random vector specification for the Hutchinson trace
                estimator.
            debias: whether to use non-uniform sampling to debias the denoising
                loss.
        """
        super(ConditionalPluginReverseSDE, self).__init__(
            base_sde=base_sde, drift_a=drift_a, T=T, vtype=vtype, debias=debias
        )
        self.grad_mask = grad_mask

    @torch.enable_grad()
    def dsm_weighted(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        w: torch.Tensor,
        clip: bool = False,
        c_min: Optional[torch.Tensor] = None,
        c_max: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Defines the weighted denoising score matching loss function.
        Input:
            x: a tensor of input x values.
            y: a tensor of corresponding y values.
            w: a tensor of corresponding weight values.
            clip: whether to clip the x values.
            c_min: minimum x value if clip is True.
            c_max: maximum x value if clip is True.
        Returns:
            The weighted denoising score matching loss term.
        """
        if self.debias:
            t_ = self.base_sde.sample_debiasing_t(
                [x.size(0)] + [1 for _ in range(x.ndim - 1)]
            )
        else:
            t_ = self.T * torch.rand(
                [x.size(0)] + [1 for _ in range(x.ndim - 1)],
                device=x.device,
                dtype=x.dtype
            )
        x_hat, target, std, g = self.base_sde.sample(t_, x, return_noise=True)

        if clip:
            c_min = c_min.repeat((x.size(0), 1))
            c_max = c_max.repeat((x.size(0), 1))
            x_hat = torch.clip(x_hat, min=c_min, max=c_max)

        a = self.a(x_hat, t_.squeeze(), y)

        loss = (w * ((a * std / g + target) ** 2)).view(x.size(0), -1)
        loss = loss * torch.unsqueeze(self.grad_mask.to(loss), dim=0)
        return torch.sum(loss, dim=-1) / 2.0

    @torch.enable_grad()
    def dsm(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Defines the unweighted denoising score matching loss function.
        Input:
            x: a tensor of input x values.
            y: a tensor of corresponding y values.
        Returns:
            The unweighted denoising score matching loss term.
        """
        return self.dsm_weighted(x, y, w=1.0)


class ConditionalDiffusionTest(DiffusionTest):
    def __init__(
        self,
        taskname: str,
        task: design_bench.task.Task,
        grad_mask: Union[np.ndarray, torch.Tensor],
        hidden_size: int = 1024,
        learning_rate: float = 1e-3,
        beta_min: float = 0.1,
        beta_max: float = 20.0,
        simple_clip: bool = False,
        activation_fn: Callable = Swish(),
        T0: float = 1.0,
        debias: bool = False,
        vtype: str = "rademacher"
    ):
        """
        Args:
            taskname: the name of the offline model-based optimization (MBO)
                task.
            task: an offline model-based optimization (MBO) task.
            grad_mask: a mask tensor indicating with dimensions of the input
                are to be optimized over.
            hidden_size: hidden size of the model. Default 1024.
            learning_rate: learning rate. Default 1e-3.
            beta_min: beta_min parameter for forward SDE model training.
            beta_max: beta_max parameter for forward SDE model training.
            activation_fn: activation function. Default Swish() function.
            T0: maximum time step.
            debias: whether to use non-uniform sampling to debias the denoising
                loss.
            vtype: random vector specification for the Hutchinson trace
                estimator.
        """
        super(ConditionalDiffusionTest, self).__init__(
            taskname=taskname,
            task=task,
            hidden_size=hidden_size,
            learning_rate=learning_rate,
            beta_min=beta_min,
            beta_max=beta_max,
            dropout_p=0.0,
            simple_clip=simple_clip,
            activation_fn=activation_fn,
            T0=T0,
            debias=debias,
            vtype=vtype
        )
        self.grad_mask = grad_mask
        if isinstance(self.grad_mask, np.ndarray):
            self.grad_mask = torch.from_numpy(self.grad_mask)
        self.grad_mask = self.grad_mask.to(self.device)

        self.gen_sde = ConditionalPluginReverseSDE(
            self.inf_sde,
            self.drift_q,
            self.T,
            grad_mask=self.grad_mask,
            vtype=self.vtype,
            debias=self.debias
        )


def ddom_train_surrogate(
    task_name: str,
    ckpt_dir: Union[Path, str] = "./checkpoints",
    batch_size: int = 128,
    num_workers: int = 0,
    seed: int = 42,
    device: str = "auto",
    num_epochs: int = 100,
    lr: float = 3e-4,
    hidden_size: int = 2048,
    dropout: float = 0.0,
) -> None:
    """
    Trains and validates a surrogate objective model for the DDOM baseline
    method for model-based optimization (MBO).
    Input:
        task_name: name of the design-bench MBO task.
        ckpt_dir: directory to saved checkpoints.
        batch_size: batch size. Default 128.
        num_workers: number of workers. Default 0.
        seed: random seed. Default 42.
        device: device. Default CPU.
        num_epochs: number of training epochs. Default 100.
        lr: learning rate. Default 0.001.
        hidden_size: hidden size of the model. Default 2048.
        dropout: dropout probability. Default 0.0
    Returns:
        None.
    """
    task = design_bench.make(task_name)
    if task.is_discrete:
        task.map_to_logits()
    if task_name == os.environ["CHEMBL_TASK"]:
        task.map_normalize_y()

    dm = RvSDataModule(
        task=task,
        val_frac=0.1,
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
        temp="90"
    )

    model = ForwardModel(
        taskname=task_name,
        task=task,
        learning_rate=lr,
        hidden_size=hidden_size,
        activation_fn=nn.LeakyReLU(negative_slope=0.2),
        dropout_p=dropout
    )

    devices = "".join(filter(str.isdigit, device))
    devices = [int(devices)] if len(devices) > 0 else "auto"
    accelerator = device.split(":")[0].lower()
    accelerator = "gpu" if accelerator == "cuda" else accelerator
    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        callbacks=[
            pl.callbacks.ModelCheckpoint(
                monitor="val_loss",
                dirpath=ckpt_dir,
                filename=f"ddom-{task_name}-surrogate"
            )
        ],
        max_epochs=num_epochs,
        logger=False
    )
    trainer.fit(model, dm)


def ddom_train(
    task_name: str,
    ckpt_dir: Union[Path, str] = "./checkpoints",
    batch_size: int = 128,
    num_workers: int = 0,
    seed: int = 42,
    device: str = "auto",
    num_epochs: int = 100,
    lr: float = 1e-3,
    hidden_size: int = 2048,
    dropout: float = 0.0,
    score_matching: bool = False,
) -> None:
    """
    DDOM training for model-based optimization (MBO).
    Input:
        task_name: name of the design-bench MBO task.
        ckpt_dir: directory to saved checkpoints.
        batch_size: batch size. Default 128.
        num_workers: number of workers. Default 0.
        seed: random seed. Default 42.
        device: device. Default CPU.
        num_epochs: number of training epochs. Default 100.
        lr: learning rate. Default 0.001.
        hidden_size: hidden size of the model. Default 2048.
        dropout: dropout probability. Default 0.0
        score_matching: whether to perform score matching. Default False.
    Returns:
        None.
    """
    task = design_bench.make(task_name)
    if task.is_discrete:
        task.map_to_logits()
    if task_name == os.environ["CHEMBL_TASK"]:
        task.map_normalize_y()

    if not os.path.isfile(
        os.path.join(ckpt_dir, f"ddom-{task_name}-surrogate.ckpt")
    ):
        ddom_train_surrogate(
            task_name, ckpt_dir, hidden_size=hidden_size, device=device
        )
        logging.info(f"Saved trained surogate model to {ckpt_dir}")

    dm = RvSDataModule(
        task=task,
        val_frac=0.1,
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
        temp="90"
    )

    if task_name in mbo.CONDITIONAL_TASKS:
        model = ConditionalDiffusionTest(
            taskname=task_name,
            task=task,
            grad_mask=task.dataset.grad_mask,
            learning_rate=lr
        )
    elif not score_matching:
        model = DiffusionTest(
            taskname=task_name,
            task=task,
            learning_rate=lr,
            dropout_p=dropout
        )
    else:
        model = DiffusionScore(
            taskname=task_name,
            task=task,
            learning_rate=lr,
            dropout_p=dropout
        )

    devices = "".join(filter(str.isdigit, device))
    devices = [int(devices)] if len(devices) > 0 else "auto"
    accelerator = device.split(":")[0].lower()
    accelerator = "gpu" if accelerator == "cuda" else accelerator
    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        callbacks=[
            pl.callbacks.ModelCheckpoint(
                monitor="elbo_estimator",
                dirpath=ckpt_dir,
                filename=f"ddom-{task_name}-{seed}"
            )
        ],
        max_epochs=num_epochs,
        logger=False
    )
    trainer.fit(model, dm)
    logging.info(f"Saved trained DDOM model to {ckpt_dir}")


@torch.no_grad()
def ddom_eval(
    task_name: str,
    ckpt_dir: Union[Path, str] = "./checkpoints",
    logging_dir: Optional[Union[Path, str]] = None,
    num_samples: int = 256,
    num_steps: int = 1000,
    hidden_size: int = 2048,
    seed: int = 42,
    device: str = "auto",
    score_matching: bool = False,
    gamma: float = 1.0
) -> None:
    """
    DDOM evaluation for model-based optimization (MBO).
    Input:
        task_name: name of the design-bench MBO task.
        ckpt_dir: directory to saved checkpoints.
        logging_dir: optional directory to save logs and results to.
        num_samples: number of samples. Default 256.
        num_steps: number of integration steps for sampling. Default 1000.
        hidden_size: hidden size of the model. Default 2048.
        seed: random seed. Default 42.
        device: device. Default CPU.
        score_matching: whether to perform score matching. Default False.
        gamma: drift parameter. Default 1.0.
    Returns:
        None.
    """
    device = get_device(device)
    task = design_bench.make(task_name)
    if task.is_discrete:
        task.map_to_logits()
    if task_name == os.environ["CHEMBL_TASK"]:
        task.map_normalize_y()

    model_ckpt = os.path.join(ckpt_dir, f"ddom-{task_name}-{seed}.ckpt")
    if task_name in mbo.CONDITIONAL_TASKS:
        model = ConditionalDiffusionTest.load_from_checkpoint(
            model_ckpt,
            taskname=task_name,
            task=task,
            grad_mask=task.dataset.grad_mask,
            map_location=device
        )
    elif not score_matching:
        model = DiffusionTest.load_from_checkpoint(
            model_ckpt, taskname=task_name, task=task, map_location=device
        )
    else:
        model = DiffusionScore.load_from_checkpoint(
            model_ckpt, taskname=task_name, task=task, map_location=device
        )
    model = model.to(device)
    model.eval()

    surrogate_ckpt = os.path.join(ckpt_dir, f"ddom-{task_name}-surrogate.ckpt")
    surrogate = ForwardModel.load_from_checkpoint(
        surrogate_ckpt,
        taskname=task_name,
        task=task,
        hidden_size=hidden_size,
        activation_fn=nn.LeakyReLU(negative_slope=0.2),
        map_location=device
    )
    surrogate.eval()

    designs, preds, scores = [], [], []
    grad_mask = None
    if hasattr(task.dataset, "grad_mask"):
        grad_mask = torch.from_numpy(task.dataset.grad_mask).to(device)

    diffusion = heun_sampler(
        task,
        model,
        num_samples=num_samples,
        num_steps=num_steps,
        device=device,
        grad_mask=grad_mask
    )
    idx = -1
    while diffusion[idx].isnan().any():
        idx -= 1
    X = diffusion[idx]
    if task.is_discrete:
        X = X.view(X.size(0), -1, task.x.shape[-1])

    designs = X.cpu().numpy()[np.newaxis, ...]
    scores = task.predict(X.cpu().numpy())[np.newaxis, ...]
    preds = surrogate.mlp(X).cpu().numpy()[np.newaxis, ...]
    if task.is_normalized_y:
        scores = task.denormalize_y(scores)

    # Save optimization results.
    if logging_dir is not None:
        os.makedirs(logging_dir, exist_ok=True)
        np.save(os.path.join(logging_dir, "solution.npy"), designs)
        np.save(os.path.join(logging_dir, "predictions.npy"), preds)
        np.save(os.path.join(logging_dir, "scores.npy"), scores)
        logging.info(f"Saved experiment results to {logging_dir}")


def main():
    args = build_args()
    seed_everything(args.seed)
    torch.set_default_dtype(torch.float32)
    ddom_train(
        args.task,
        hidden_size=args.hidden_size,
        device=args.device,
        seed=args.seed
    )
    ddom_eval(
        args.task,
        logging_dir=args.logging_dir,
        hidden_size=args.hidden_size,
        num_samples=args.budget,
        device=args.device,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
