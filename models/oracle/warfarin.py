"""
Estimates the target weekly warfarin dose based on a pharmacogenetic or
clinical dosing algorithm.

Author(s):
    Michael Yao @michael-s-yao

Citation(s):
    [1] The International Warfarin Pharmacogenetics Consortium. Estimation of
        the warfarin dose with clinical and pharmacogenetic data. N Engl J Med
        360:753-64. (2009). https://doi.org/10.1056/NEJMoa0809329

Licensed under the MIT License. Copyright University of Pennsylvania 2023.
"""
import numpy as np
import pandas as pd
from typing import Any, Dict, Union


class WarfarinDosingOracle:
    """
    Warfarin weekly dose estimator using one of the linear regression models
    reported by IWPC. (NEJM 2009).
    """

    def __init__(self, use_pharmacogenetic_algorithm: bool = True):
        """
        Args:
            use_pharmacogenetic_algorithm: whether to use the pharmacogenetic
                dosing algorithm instead of the clinical dosing algorithm
                presented by IWPC. (NEJM 2009). Default True.
        """
        self.use_pharmacogenetic_algorithm = use_pharmacogenetic_algorithm
        self._enzyme_inducers = (
            "Carbamazepine (Tegretol)",
            "Phenytoin (Dilantin)",
            "Rifampin or Rifampicin"
        )
        self._cyp2c9_consensus = [
            "CYP2C9 consensus_*1/*1",
            "CYP2C9 consensus_*1/*2",
            "CYP2C9 consensus_*1/*3",
            "CYP2C9 consensus_*2/*2",
            "CYP2C9 consensus_*2/*3",
            "CYP2C9 consensus_*3/*3",
        ]
        if self.use_pharmacogenetic_algorithm:
            self.algorithm = self._pharmacogenetic_dosing_algorithm
        else:
            self.algorithm = self._clinical_dosing_algorithm

    def __call__(self, X: Union[pd.Series, pd.DataFrame]) -> np.ndarray:
        """
        Predicts the stable weekly dose of warfarin in mg according to one of
        either the pharmacogenetic or clinical dosing algorithms presented in
        IWPC. (NEJM 2009).
        Input:
            X: a DataFrame or Series of patient data.
        Returns:
            The predicted stable weekly dose of warfarin for the patient(s).
        """
        return self._dosing_algorithm(X, self.algorithm)

    def _dosing_algorithm(
        self, X: Union[pd.Series, pd.DataFrame], algorithm: Dict[Any, float]
    ) -> np.ndarray:
        """
        Predicts the stable weekly dose of warfarin in mg according to one of
        either the pharmacogenetic or clinical dosing algorithms presented in
        IWPC. (NEJM 2009).
        Input:
            X: a DataFrame or Series of patient data.
            algorithm: one of [`self._pharmacogenetic_dosing_algorithm`,
                `self._clinical_dosing_algorithm`].
        Returns:
            The predicted stable weekly dose of warfarin for the patient(s).
        """
        dose = 0.0 if isinstance(X, pd.Series) else np.zeros(len(X))
        for var, m in algorithm.items():
            if var == self._enzyme_inducers:
                dose += m * np.any(
                    X[list(self._enzyme_inducers)].to_numpy(), axis=-1
                )
            elif var == "bias":
                dose += m
            elif var not in X.columns:
                continue
            else:
                dose += X[var] * m
        return np.square(np.array(dose))

    @property
    def _pharmacogenetic_dosing_algorithm(self) -> Dict[Any, float]:
        """
        Returns the pharmacogenetic dosing algorithm model parameters.
        Input:
            None.
        Returns:
            A dictionary mapping patient attributes to their respective
            multiplicative weights in the linear regression dosing model.
        """
        return {
            "bias": 5.6044,
            "Age": -0.2614,
            "Height (cm)": 0.0087,
            "Weight (kg)": 0.0128,
            "Imputed VKORC1_A/G": -0.8677,
            "Imputed VKORC1_A/A": -1.6974,
            "Imputed VKORC1_Unknown": -0.4854,
            "CYP2C9 consensus_*1/*2": -0.5211,
            "CYP2C9 consensus_*1/*3": -0.9357,
            "CYP2C9 consensus_*2/*2": -1.0616,
            "CYP2C9 consensus_*2/*3": -1.9206,
            "CYP2C9 consensus_*3/*3": -2.3312,
            "CYP2C9 consensus_unknown": -0.2188,
            "Race (OMB)_Asian": -0.1092,
            "Race (OMB)_Black or African American": -0.2760,
            "Race (OMB)_Unknown": -0.1032,
            self._enzyme_inducers: 1.1816,
            "Amiodarone (Cordarone)": -0.5503
        }

    @property
    def _clinical_dosing_algorithm(self) -> Dict[Any, float]:
        """
        Returns the clinical dosing algorithm model parameters.
        Input:
            None.
        Returns:
            A dictionary mapping patient attributes to their respective
            multiplicative weights in the linear regression dosing model.
        """
        return {
            "bias": 4.0376,
            "Age": -0.2546,
            "Height (cm)": 0.0118,
            "Weight (kg)": 0.0134,
            "Race (OMB)_Asian": -0.6752,
            "Race (OMB)_Black or African American": 0.4060,
            "Race (OMB)_Unknown": 0.0443,
            self._enzyme_inducers: 1.2799,
            "Amiodarone (Cordarone)_1.0": -0.5695
        }
