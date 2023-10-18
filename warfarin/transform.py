"""
Defines a transform class to normalize and reconstruct vectors.

Author(s):
    Michael Yao @michael-s-yao

Licensed under the MIT License. Copyright University of Pennsylvania 2023.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import Union


class PowerNormalizeTransform:
    """Transforms a vector using a power law followed by normalization."""

    def __init__(
        self,
        X: Union[pd.DataFrame, pd.Series, np.ndarray],
        p: Union[float, int] = 1,
        key: str = "Therapeutic Dose of Warfarin"
    ):
        """
        Args:
            X: a vector to fit the transform to.
            p: power to raise X's elements to prior to fitting the transform.
            key: column to transform. Must be provided if X is a DataFrame.
        """
        self.scaler = StandardScaler()
        self.p = p
        self.key = key
        self.z_atol = 10.0
        if isinstance(X, pd.DataFrame):
            X = X[self.key].to_numpy()
        elif isinstance(X, pd.Series):
            X = X.to_numpy()
        X = np.squeeze(X)
        self.scaler = self.scaler.fit(np.power(X, self.p)[:, np.newaxis])

    def __call__(
        self, X: Union[pd.DataFrame, pd.Series, np.ndarray]
    ) -> Union[pd.DataFrame, np.ndarray]:
        """
        Normalizes a vector of raw values using the fitted transform.
        Input:
            X: a vector of raw values.
        Returns:
            A vector of the normalized values.
        """
        return self.normalize(X)

    def normalize(
        self, X: Union[pd.DataFrame, pd.Series, np.ndarray]
    ) -> Union[pd.DataFrame, np.ndarray]:
        """
        Normalizes a vector of raw values using the fitted transform.
        Input:
            X: a vector of raw values.
        Returns:
            A vector of the normalized values.
        """
        if self._is_normalized(X):
            raise ValueError(
                "Attempting to normalize already normalized values!"
            )
        if isinstance(X, pd.DataFrame):
            Xp = np.squeeze(np.power(X[self.key].to_numpy(), self.p))
            z = X.copy()
            z[self.key] = self.scaler.transform(Xp[:, np.newaxis])
            return z
        elif isinstance(X, pd.Series):
            X = X.to_numpy()
        X = np.squeeze(X)[:, np.newaxis]
        return self.scaler.transform(np.power(X, self.p))

    def invert(
        self, z: Union[pd.DataFrame, pd.Series, np.ndarray]
    ) -> Union[pd.DataFrame, np.ndarray]:
        """
        Reconstructs a vector of raw values using input normalized values.
        Input:
            z: a vector of normalized values.
        Returns:
            A vector of the raw values.
        """
        if not self._is_normalized(z):
            raise ValueError(
                "Attempting to invert already reconstructed values!"
            )
        if isinstance(z, pd.DataFrame):
            X = z.copy()
            z = np.squeeze(z[self.key].to_numpy())[:, np.newaxis]
            Xp = self.scaler.inverse_transform(z)
            X[self.key] = np.power(Xp, 1.0 / self.p)
            return X
        elif isinstance(z, pd.Series):
            z = z.to_numpy()
        z = np.squeeze(z)[:, np.newaxis]
        return np.squeeze(
            np.power(self.scaler.inverse_transform(z), 1.0 / self.p), axis=-1
        )

    def _is_normalized(
        self, X: Union[pd.DataFrame, pd.Series, np.ndarray]
    ) -> bool:
        """
        Returns whether an input vector is a vector of normalized values.
        Input:
            X: a vector of potentially normalized values.
        Returns:
            Whether the vector is normalized or not.
        """
        if isinstance(X, pd.DataFrame):
            X = X[self.key].to_numpy()
        elif isinstance(X, pd.Series):
            X = X.to_numpy()
        X = np.squeeze(X)
        if not -self.z_atol <= np.mean(X) <= self.z_atol:
            return False
        return -self.z_atol <= np.std(X, ddof=1) - 1.0 <= self.z_atol
