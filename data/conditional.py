"""
Defines the ConditionalContinuousDataset for conditional model-based
optimization (MBO) in which only a few of the input design dimensions may be
optimized over.

Author(s):
    Michael Yao @michael-s-yao

Licensed under the MIT License. Copyright University of Pennsylvania 2023.
"""
import numpy as np
from design_bench.datasets.continuous_dataset import ContinuousDataset
from design_bench.disk_resource import DiskResource
from typing import Callable, Optional, Sequence, Union


class ConditionalContinuousDataset(ContinuousDataset):
    """
    Defines a continuous dataset base class for conditional model-based
    optimization (MBO) in which only a few of the input design dimensions may
    be optimized over.
    """

    def __init__(
        self,
        x_shards: Union[np.ndarray, DiskResource],
        y_shards: Union[np.ndarray, DiskResource],
        grad_mask: np.ndarray,
        column_names: Optional[Sequence[str]] = None,
        internal_batch_size: int = 32,
        is_normalized_x: bool = False,
        is_normalized_y: bool = False,
        max_samples: Optional[int] = None,
        distribution: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        max_percentile: float = 100.0,
        min_percentile: float = 0.0
    ):
        """
        Args:
            x_shards: a single shard or list of shards representing the
                observed design dimensions in the MBO dataset that will be
                optimized over.
            grad_mask: a mask of input design dimensions that can be optimized
                over. The mask should be True for dimensions that will be
                optimized over and False for frozen condition dimensions.
            column_names: an optional list of the design dimension names.
            y_shards: a single shard or list of shards representing the
                objective values of the MBO dataset.
            internal_batch_size: the number of samples per batch to use when
                computing normalization statistics of the dataset and while
                relabeling the prediction values of the dataset.
            is_normalized_x: whether the input designs are normalized.
            is_normalized_y: whether the input objective values are normalized.
            max_samples: the maximum number of samples to include in the
                visible dataset.
            distribution: a function that accepts an array of the ranks of
                designs as input and returns the probability to sample each
                according to the specified distribution.
            max_percentile: the percentile of prediction values above which are
                hidden from access by members outside the class.
            min_percentile: the percentile of prediction values below which are
                hidden from access by members outside the class.
        """
        super(ConditionalContinuousDataset, self).__init__(
            x_shards=x_shards,
            y_shards=y_shards,
            internal_batch_size=internal_batch_size,
            is_normalized_x=is_normalized_x,
            is_normalized_y=is_normalized_y,
            max_samples=max_samples,
            distribution=distribution,
            max_percentile=max_percentile,
            min_percentile=min_percentile
        )
        self.grad_mask = grad_mask
        self.column_names = column_names
