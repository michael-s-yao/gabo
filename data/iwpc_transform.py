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
            column = data[transform.column_name].to_numpy()
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
