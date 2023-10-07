"""
Data transformer to transform continuous columns in a tabular dataset.

Author(s):
    Michael Yao

Citation(s):
    [1] Xu Lei, Skoularidou M, Cuesta-Infante A, Veeramachaneni K. Modeling
        tabular data using conditional GAN. Proc NeurIPS. (2019).
        https://doi.org/10.48550/arXiv.1907.00503
    [2] CTGAN Github repo from @sdv-dev at https://github.com/sdv-dev/CTGAN

Licensed under the MIT License. Copyright University of Pennsylvania 2023.
"""
import numpy as np
import pandas as pd
import torch
from rdt.transformers import ClusterBasedNormalizer
from typing import Callable, NamedTuple, Sequence


class ColumnTransform(NamedTuple):
    column_name: str
    transform: Callable[[pd.DataFrame], pd.DataFrame]
    out_dim: int


class TGANContinuousDataTransform:
    """
    Models continuous columns with a BayesianGMM and normalizes them to a
    scalar between [-1, 1] and a vector.
    """

    def __init__(
        self, max_clusters: int = 10, weight_threshold: float = 0.005
    ):
        """
        Args:
            max_clusters: maximum number of Gaussian distributions in
                BayesianGMM.
            weight_threshold: weight threshold for a Gaussian distribution to
                be kept.
        """
        self._max_clusters = max_clusters
        self._weight_threshold = weight_threshold
        self._transforms = []
        self.out_dim = 0

    def _fit_continuous(self, data: pd.DataFrame) -> ColumnTransform:
        """
        Train a Bayesian GMM for columns containing continuous variables.
        Input:
            data: input dataframe containing a column.
        Returns:
            A ColumnTransform NamedTuple containing the trained transform.
        """
        gm = ClusterBasedNormalizer(
            max_clusters=min(len(data), self._max_clusters),
            weight_threshold=self._weight_threshold
        )
        gm.fit(data, data.columns[0])
        num_components = sum(gm.valid_component_indicator)

        return ColumnTransform(data.columns[0], gm, 1 + num_components)

    def fit(
        self, data: pd.DataFrame, continuous_columns: Sequence[str]
    ) -> None:
        """
        Fits individual Bayesian GMMs for columns containing continuous
        variables.
        Input:
            data: input dataframe containing the patient dataset.
            continuous_columns: a list of all of the continuous columns.
        Return:
            None.
        """
        self.continuous_columns = continuous_columns
        for col in data.columns:
            if col not in continuous_columns:
                continue
            col_transform = self._fit_continuous(data[[col]])
            self._transforms.append(col_transform)
            self.out_dim += col_transform.out_dim
        return

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms continuous variables continued within a dataset according
        to the fitted transforms.
        Input:
            data: input dataframe containing the patient dataset to transform.
        Returns:
            Transformed dataframe.
        """
        for transform in self._transforms:
            column_name = transform.column_name
            column = data[column_name].to_numpy()
            data = data.assign(**{column_name: column})
            data = transform.transform.transform(data)
            one_hot_encodings = {
                f"{column_name}.component_{int(idx)}": np.zeros(
                    len(data), dtype=int
                )
                for idx in range(transform.out_dim)
            }
            for i, idx in enumerate(data[f"{column_name}.component"]):
                key = f"{column_name}.component_{int(idx)}"
                one_hot_encodings[key][i] = 1

            data = data.assign(**one_hot_encodings)
            data = data.drop([f"{column_name}.component"], axis=1)

        return data

    def invert(
        self, data: torch.Tensor, attributes: Sequence[str]
    ) -> torch.Tensor:
        """
        Inverts the transform according to the fitted BayesianGMM transforms.
        Input:
            data: input data contataing the transformed patient dataset to
                invert.
        Returns:
            The inverted tensor.
        """
        els = []
        for attr in attributes:
            attr = attr.replace(".normalized", "").split(".component")[0]
            if attr in els:
                continue
            if (
                attr.endswith("_0.0") and
                attr.replace("_0.0", "_1.0") in attributes
            ):
                els.append(attr.replace("_0.0", ""))
            elif not attr.endswith("_1.0"):
                els.append(attr)
        raw_data = torch.zeros((data.size(dim=0), len(els))).to(data)

        # Invert BayesianGMM transforms.
        inverted = {}
        for transform in self._transforms:
            column_name = transform.column_name
            normalized = data[:, attributes.index(f"{column_name}.normalized")]
            component_start = attributes.index(f"{column_name}.component_0")
            component_end = component_start + transform.out_dim
            component = torch.argmax(
                data[:, component_start:component_end], dim=-1
            )
            means = transform.transform._bgm_transformer.means_
            stds = np.sqrt(transform.transform._bgm_transformer.covariances_)
            means = torch.tensor(np.squeeze(means)).to(data)
            stds = torch.tensor(np.squeeze(stds)).to(data)
            mean_t = means[component].to(data)
            std_t = stds[component].to(data)
            inverted[column_name] = mean_t + (
                normalized * transform.transform.STD_MULTIPLIER * std_t
            )

        # Invert discrete variables with only two values.
        for attr in attributes:
            if (
                not attr.endswith("_0.0") or
                attr.replace("_0.0", "_1.0") not in attributes
            ):
                continue
            attr = attr.replace("_0.0", "")
            f_idx = attributes.index(f"{attr}_0.0")
            t_idx = attributes.index(f"{attr}_1.0") + 1
            inverted[attr] = torch.argmax(data[:, f_idx:t_idx], dim=-1)

        raw_data = torch.zeros((data.size(dim=0), len(els))).to(data)
        for i, attr in enumerate(els):
            if attr not in inverted.keys():
                raw_data[:, i] = data[:, attributes.index(attr)]
                continue
            raw_data[:, i] = inverted[attr]
        return raw_data
