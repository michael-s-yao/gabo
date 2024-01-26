#!/usr/bin/env python3
"""
Main driver program for the BONET baseline MBO method.

Author(s):
    Michael Yao @michael-s-yao

Citation(s):
    [1] Krishnamoorthy S, Mashkaria S, Grover A. Diffusion models for black-
        box optimization. Proc ICML 734:17842-857. (2023).
        https://dl.acm.org/doi/10.5555/3618408.3619142

Adapted from the bonet GitHub repo by @siddarthk97 at https://github.com/
siddarthk97/bonet. Specifically, the relevant source code files are:
    [1] bonet/scripts/sorted_binning.py
    [2] bonet/scripts/train_desbench_new.py
    [3] bonet/scripts/test_desbench_new.py

Licensed under the MIT License. Copyright University of Pennsylvania 2023.
"""
import argparse
import logging
import numpy as np
import pickle
import os
import sys
import torch
from math import ceil
from pathlib import Path
from typing import NamedTuple, Optional, Tuple, Union

sys.path.append(".")
sys.path.append("bonet")
import mbo  # noqa
import design_bench
from mbo.run_gabo import load_vae_and_surrogate_models
from bonet.mingpt.model import GPT, GPTConfig
from bonet.mingpt.model_discrete_new import GPTDiscrete
from bonet.mingpt.model_discrete_new import GPTConfig as GPTConfigDiscrete
from bonet.mingpt.trainer import Trainer, TrainerConfig
from models.logger import DummyLogger
from helpers import seed_everything, get_device


def build_args() -> argparse.Namespace:
    """
    Defines the experimental arguments for offline MBO baseline method
    evaluation.
    Input:
        None.
    Returns:
        A namespace containing the experimental argument values.
    """
    parser = argparse.ArgumentParser(description="BONET Baseline Experiments")

    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=[task.task_name for task in design_bench.registry.all()],
        help="The name of the design-bench task for the experiment."
    )
    parser.add_argument(
        "--context-length",
        type=int,
        default=40,
        help="Context length for BONET models. Default 40."
    )
    parser.add_argument(
        "--logging-dir",
        type=str,
        default=None,
        help="Logging directory to save optimization results to."
    )
    parser.add_argument(
        "--cond-rtg",
        type=float,
        default=8.0,
        help="Regret budget for new design generation."
    )
    parser.add_argument(
        "--val-init-length",
        type=int,
        default=64,
        help="Prefix trajectory length for trajectory generation."
    )
    parser.add_argument(
        "--budget", type=int, default=256, help="Query budget. Default 256."
    )
    parser.add_argument(
        "--ckpt-dir",
        type=str,
        default="./checkpoints/bonet",
        help="Directory to save model checkpoints to."
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed. Default 42."
    )
    parser.add_argument(
        "--device", type=str, default="auto", help="Device. Default `auto`."
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="both",
        choices=("fit", "eval", "both"),
        help="Specify model training or evaluation. Default both."
    )

    return parser.parse_args()


class TrajectoryDataset(NamedTuple):
    trajectories: torch.Tensor
    trajectory_values: torch.Tensor
    pointwise_regret: torch.Tensor
    cumulative_regret_to_go: torch.Tensor
    timesteps: torch.Tensor
    optima: torch.Tensor


class BONETTrajectories:
    """
    Implements the sorted binning script from https://github.com/siddarthk97/
    bonet/scripts/sorted_binning.py to construct synthetic optimization
    trajectory data BONET training as described by Mashkaria et al. (2023).
    """

    def __init__(
        self,
        task: design_bench.task.Task,
        task_name: str,
        cache_dir: Optional[Union[Path, str]] = "./bonet/generated_datasets",
        num_bins: int = 64,
        traj_len: int = 128,
        num_train_trajectories: int = 800,
        num_eval_trajectories: int = 128,
        seed: int = 42
    ):
        """
        Args:
            task: a model-based optimization (MBO) task.
            task_name: name of the MBO task.
            cache_dir: an optional path to a folder to save trajectory data.
            num_bins: number of bins to construct from the dataset.
            traj_len: length of the sampling trajectories to construct.
            num_train_trajectories: number of training trajectories to build.
            num_eval_trajectories: number of validation trajectories to build.
            seed: random seed. Default 42.
        """
        self.task = task
        self.task_name = task_name
        self.cache_dir = cache_dir
        self.num_bins = num_bins
        self.traj_len = traj_len
        self.num_train_trajectories = num_train_trajectories
        self.num_eval_trajectories = num_eval_trajectories
        self.seed = seed
        self.rng = np.random.RandomState(self.seed)

        self.K = 0.03 * self.task.x.shape[0]
        self.values = (self.task.y - self.task.dataset.y.min()) / (
            self.task.dataset.y.max() - self.task.dataset.y.min()
        )
        self.values = np.squeeze(self.values, axis=-1)

        self.regrets = self.values.max() - self.values
        self.dataset = np.hstack([self.task.x, self.regrets.reshape(-1, 1)])

    def _sampling_bins(self) -> np.ndarray:
        """
        Constructs the sampling bins where each bin contains the indices of
        the datums that fall under that objective value bin.
        Input:
            None.
        Returns:
            The sampling bins.
        """
        bin_len = (self.regrets.max() - self.regrets.min()) / self.num_bins
        bins = [[] for i in range(self.num_bins)]
        for i, y in enumerate(self.values):
            bin_idx = min(
                self.num_bins - ceil(y.item() / bin_len), self.num_bins - 1
            )
            bins[bin_idx].append(i)
        return bins

    def _score_bins(self) -> np.ndarray:
        """
        Scores each of the sampling bins according to Equation (2) of
        Mashkaria et al. (2023).
        Input:
            None.
        Returns:
            The number of samples to sample from each bin to form the final
            trajectories.
        """
        bins = self._sampling_bins()
        tau = self.values.max() - np.percentile(self.regrets, 90)
        scores = []
        for b in range(len(bins)):
            y_bi = (1.0 / self.num_bins) * (
                (len(bins) - b - 0.5) * (self.values.max() - self.values.min())
            )
            scores.append(
                (len(bins[b]) / (len(bins[b]) + self.K)) * np.exp(
                    -1.0 * abs(self.values.max() - y_bi) / tau
                )
            )
        scores = np.array(scores)
        num_samples = np.round(self.traj_len * (scores / np.sum(scores)))
        num_samples = num_samples.astype(np.int32)
        num_samples[0] += (self.traj_len - np.sum(num_samples))
        return num_samples

    def sort_sample(self, stage: str = "fit") -> TrajectoryDataset:
        """
        Implements the SORT-SAMPLE algorithm from Mashkaria et al. (2023) to
        construct the input data for the BONET autoregressive model.
        Input:
            stage: one of [`fit`, `eval`].
        Returns:
            A TrajectoryDataset of the relevant data.
        """
        num_traj = self.num_train_trajectories
        if stage.lower() != "fit":
            num_traj = self.num_eval_trajectories

        # Try loading from the cache first.
        if self.cache_dir is not None:
            cache_path = os.path.join(
                self.cache_dir,
                f"{self.task_name}_{num_traj}x{self.traj_len}_{stage}.pkl"
            )
            if os.path.isfile(cache_path):
                with open(cache_path, "rb") as f:
                    data = pickle.load(f)
                logging.info(f"Loaded BONET trajectory data from {cache_path}")
                return data

        bins, num_samples = self._sampling_bins(), self._score_bins()
        trajectories, trajectory_vals = [], []
        for i in range(num_traj):
            points, ys = [], []
            for b in range(self.num_bins):
                idxs = self.rng.choice(bins[b], num_samples[b], replace=True)
                idxs = idxs.astype(np.int32)
                points.append(self.dataset[idxs, :-1])
                ys.append(self.values.max() - self.dataset[idxs, -1])
            trajectory, ys = np.vstack(points), np.concatenate(ys)

            idxs = np.argsort(ys)
            trajectory, ys = trajectory[idxs, :], ys[idxs]

            trajectories.append(trajectory[np.newaxis, ...])
            trajectory_vals.append(ys[np.newaxis, ...])
        trajectories = np.concatenate(trajectories, axis=0)
        trajectory_vals = np.concatenate(trajectory_vals, axis=0)

        pointwise_regret = self.values.max() - trajectory_vals
        cumulative_regret_to_go = np.flip(
            np.cumsum(np.flip(pointwise_regret, axis=-1), axis=-1), axis=-1
        )
        timesteps = np.arange(self.traj_len)[:, np.newaxis]
        timesteps = timesteps.repeat(self.num_train_trajectories, 1).T

        data = TrajectoryDataset(
            torch.from_numpy(trajectories),
            torch.from_numpy(trajectory_vals),
            torch.from_numpy(pointwise_regret),
            torch.from_numpy(cumulative_regret_to_go.copy()),
            torch.from_numpy(timesteps),
            torch.tensor(self.values.max())
        )
        if self.cache_dir is not None:
            with open(cache_path, "wb") as f:
                pickle.dump(data, f)
            logging.info(f"Saved BONET trajectory data to {cache_path}")
        return data


class PointRegretDataset(torch.utils.data.Dataset):
    """
    Implements the PointRegretDataset from https://github.com/siddarthk97/
    bonet/scripts/train_desbench.py to construct a dataset to train an
    autoregressive model to learn optimization trajectories as described by
    Mashkaria et al. (2023).
    """

    def __init__(
        self,
        task: design_bench.task.Task,
        data: TrajectoryDataset,
        block_size: int
    ):
        """
        Args:
            task: a model-based optimization (MBO) task.
            block_size: trajectory lengths of the data in the dataset.
            data: a dataset of synthetic optimization trajectories.
        """
        self.is_discrete = task.is_discrete
        self.data = data
        self.block_size = block_size
        self.vocab_size = 1 if not self.is_discrete else task.num_classes
        self.num_trajectories, self.traj_len, _ = self.data.trajectories.size()

    def __len__(self) -> int:
        """
        Returns the total number of datums in the dataset.
        Input:
            None.
        Returns:
            The total number of datums in the dataset.
        """
        return (self.num_trajectories * self.traj_len) - self.block_size

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor]:
        """
        Returns a specified datum from the dataset.
        Input:
            idx: the index of the dataum from the dataset to retrieve.
        Returns:
            The specified datum from the dataset.
        """
        block_size = self.block_size // 2
        traj_idx = idx // self.traj_len
        sidx = idx - traj_idx * self.traj_len
        if sidx + block_size > self.traj_len:
            sidx = self.traj_len - block_size
        eidx = sidx + block_size
        return (
            self.data.trajectories[traj_idx, sidx:eidx],
            self.data.trajectories[traj_idx, sidx:eidx],
            torch.unsqueeze(
                self.data.cumulative_regret_to_go[traj_idx, sidx:eidx], dim=-1
            ),
            self.data.timesteps[traj_idx, sidx:(sidx + 1)].unsqueeze(dim=-1)
        )


class ForwardModel:
    """
    Implements a surrogate objective model using the BONET API. To avoid
    needing to train another surrogate, the COMBO-SCR-associated surrogate is
    used here instead.
    """

    def __init__(
        self,
        task: design_bench.task.Task,
        task_name: str,
        device: torch.device = torch.device("cpu"),
        **kwargs
    ):
        """
        Args:
            task: an offline model-based optimization (MBO) task.
            task_name: the name of the offline MBO task.
            device: device. Default CPU.
        """
        self.task, self.task_name, self.device = task, task_name, device
        self.vae, self.surrogate = load_vae_and_surrogate_models(
            task=self.task, task_name=self.task_name, **kwargs
        )
        self.vae = self.vae.to(self.device)
        self.surrogate = self.surrogate.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the surrogate objective model.
        Input:
            x: an input design or batch of designs.
        Returns:
            The predicted objective values associated with the input.
        """
        z, _, _ = self.vae.encode(x.to(self.device))
        if z.ndim > 2:
            z = z.flatten(start_dim=1)
        return self.surrogate(z)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the surrogate objective model.
        Input:
            x: an input design or batch of designs.
        Returns:
            The predicted objective values associated with the input.
        """
        return self.forward(x)

    def regret(
        self, x: Union[np.ndarray, torch.Tensor]
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Computes the regret using the surrogate objective given an input
        batch of designs.
        Input:
            x: an input batch of designs.
        Returns:
            The predicted regret for the input designs.
        """
        is_np = isinstance(x, np.ndarray)
        x = torch.from_numpy(x) if is_np else x
        y = self(x)
        y = y.detach().cpu().numpy() if is_np else y
        return self.task.y.max() - y


def main():
    args = build_args()
    seed_everything(args.seed)
    device = get_device(args.device)
    torch.set_default_dtype(torch.float32)

    task = design_bench.make(args.task)

    dataset = BONETTrajectories(task, args.task)
    train = PointRegretDataset(
        task, dataset.sort_sample("fit"), 2 * args.context_length
    )
    val = dataset.sort_sample("eval")

    config = GPTConfigDiscrete if task.is_discrete else GPTConfig
    # Default hyperparameters are the same as those in the BONET source code
    # https://github.com/siddarthk97/bonet/scripts/train_desbench_new.py
    default_bonet_config = {
        "input_dim": task.input_shape[0],
        "n_layer": 8,
        "n_head": 16,
        "n_embd": 128,
        "max_timestep": 128
    }
    model = GPTDiscrete if task.is_discrete else GPT
    model = model(
        config(train.vocab_size, train.block_size, **default_bonet_config)
    )

    ckpt_path = os.path.join(args.ckpt_dir, str(args.seed))
    os.makedirs(args.ckpt_dir, exist_ok=True)
    if args.mode in ["train", "both"] and not os.path.isfile(ckpt_path):
        # Default hyperparameters are equal to those in the BONET source code
        # https://github.com/siddarthk97/bonet/scripts/train_desbench_new.py
        tconfig = {
            "max_epochs": 50,
            "batch_size": 128,
            "learning_rate": 1e-4,
            "lr_decay": False,
            "warmup_tokens": 512 * 20,
            "final_tokens": (2 * len(train)) * (2 * args.context_length),
            "num_workers": 0,
            "seed": args.seed,
            "max_timestep": 128,
            "ckpt_path": ckpt_path
        }
        trainer = Trainer(
            model,
            train,
            PointRegretDataset(task, val, 2 * args.context_length),
            TrainerConfig(**tconfig),
            add_noise=False
        )
        trainer.train(writer=DummyLogger())
    if args.mode == "train":
        return

    model.load_state_dict(torch.load(f"{ckpt_path}_best"))
    model = model.to(device)

    all_X, all_regrets, all_y_gt = [], [], []
    for _ in range(
        ceil(args.budget // (train.traj_len - args.val_init_length))
    ):
        traj_idx = dataset.rng.randint(0, dataset.num_eval_trajectories)
        val_pr = val.pointwise_regret[traj_idx, :args.val_init_length]
        points, regrets = model.evaluate(
            rtg=args.cond_rtg,
            unroll_length=default_bonet_config["max_timestep"],
            function=ForwardModel(task, args.task),
            device=str(device),
            update_regret=True,
            initial_points=val.trajectories[traj_idx, :args.val_init_length],
            initial_rtgs=torch.flip(
                torch.cumsum(torch.flip(val_pr, dims=[0]), dim=0), dims=[0]
            )
        )
        all_X.append(np.array([points]))
        all_regrets.append(np.array([regrets]))
        all_y_gt.append(task.predict(np.array(points))[np.newaxis, ...])
    all_X = np.concatenate(all_X, axis=0)
    all_y = -1.0 * np.concatenate(all_regrets, axis=0)[..., np.newaxis]
    all_y_gt = np.concatenate(all_y_gt, axis=0)

    # Save optimization results.
    if args.logging_dir is not None:
        os.makedirs(args.logging_dir, exist_ok=True)
        np.save(os.path.join(args.logging_dir, "solution.npy"), all_X)
        np.save(os.path.join(args.logging_dir, "predictions.npy"), all_y)
        np.save(os.path.join(args.logging_dir, "scores.npy"), all_y_gt)
        logging.info(f"Saved experiment results to {args.logging_dir}")


if __name__ == "__main__":
    main()
