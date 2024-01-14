"""
Defines the oracle objective for the Branin toy task.

Author(s):
    Michael Yao

Citation(s):
    [1] Branin FH. Widely convergent method for finding multiple solutions of
        simultaneous nonlinear equations. IBM J Res and Dev 16(5):504-22.
        (1972). https://doi.org/10.1147/rd.165.0504
    [2] Deng L. The MNIST database of handwritten digit images for machine
        learning research. IEEE Sig Proc Magazine 29(6):141-2. (2012).
        https://doi.org/10.1109/MSP.2012.2211477
    [3] Brown N, Fiscato M, Segler MHS, Vaucher AC. GuacaMol: Benchmarking
        models for de novo molecular design. J Chem Inf Model 59(3):1096-08.
        (2019). https://doi.org/10.1021/acs.jcim.8b00839
    [4] The International Warfarin Pharmacogenetics Consortium. Estimation of
        the warfarin dose with clinical and pharmacogenetic data. N Engl J Med
        360:753-64. (2009). https://doi.org/10.1056/NEJMoa0809329

Licensed under the MIT License. Copyright University of Pennsylvania 2023.
"""
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import sys
import torch
import torch.nn as nn
from botorch.test_functions.synthetic import Branin
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import Crippen
from rdkit.Chem import rdmolops
from guacamol import standard_benchmarks as SB
from typing import Any, Dict, Optional, Sequence, Tuple, Union

sys.path.append(".")
from helpers import plot_config
import models.sascorer as sascorer


class BraninOracle(nn.Module):
    def __init__(self, negate: bool = True):
        """
        Args:
            negate: whether to return the negative of the Branin function.
        """
        super().__init__()
        self.negate = negate
        self.oracle = Branin(negate=self.negate)

    def forward(
        self, X: Union[np.ndarray, torch.Tensor],
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Computes the negative 2D Brainin function at a specified point(s).
        Input:
            X: a coordinate or batch of coordinates.
        Returns:
            The function value(s) at X.
        """
        X = X[np.newaxis] if X.ndim == 1 else X
        is_np = isinstance(X, np.ndarray)
        X = torch.from_numpy(X) if is_np else X
        y = self.oracle(X).to(X)
        return y.detach().cpu().numpy() if is_np else y

    @property
    def optima(self) -> Sequence[Tuple[float]]:
        """
        Returns the optimal points according to the oracle.
        Input:
            None.
        Returns:
            A list of (x1, x2) pairs corresponding to the optimal points.
        """
        return [(-np.pi, 12.275), (np.pi, 2.275), (9.42478, 2.475)]

    def show(self, savepath: Optional[Union[Path, str]] = None) -> None:
        """
        Generates a contour plot of the objective function.
        Input:
            savepath: optional path to save the plot to.
        Returns:
            None.
        """
        plot_config(fontsize=20)
        bounds = self.oracle.bounds.detach().cpu().numpy()
        x1_range, x2_range = bounds[:, 0], bounds[:, 1]
        x1 = np.linspace(min(x1_range), max(x1_range), num=1000)
        x2 = np.linspace(min(x2_range), max(x2_range), num=1000)
        x1, x2 = np.meshgrid(x1, x2)
        plt.figure()
        plt.contour(x1, x2, self(x1, x2), levels=100, cmap="jet")
        plt.xlabel(r"$x_1$")
        plt.ylabel(r"$x_2$")
        plt.colorbar()
        plt.scatter(
            [x1 for x1, _ in self.optima],
            [x2 for _, x2 in self.optima],
            color="k",
            marker=(5, 1),
            label="Maxima"
        )
        plt.legend(loc="lower right")
        if savepath is None:
            plt.show()
        else:
            plt.savefig(
                savepath, dpi=600, transparent=True, bbox_inches="tight"
            )
        plt.close()


class MNISTOracle(nn.Module):
    def __init__(self):
        """
        Args:
            None.
        """
        super().__init__()

    def forward(
        self, X: Union[np.ndarray, torch.Tensor],
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Computes the energy of a specified MNIST image or batch of images.
        Input:
            X: an image or batch of images.
        Returns:
            The squared L2 norm of X.
        """
        X = X[np.newaxis] if X.ndim % 2 else X
        X = X.reshape(X.shape[0], -1)
        if isinstance(X, np.ndarray):
            return np.mean(np.square(np.clip(X, 0.0, 1.0)), axis=-1).astype(
                X.dtype
            )
        return torch.mean(torch.square(torch.clamp(X, 0.0, 1.0)), dim=-1).to(
            X.dtype
        )


class MoleculeOracle:
    def __init__(self, objective: Optional[str] = None):
        self.guacamol_objs = {
            "med1": SB.median_camphor_menthol(),
            "pdop": SB.perindopril_rings(),
            "adip": SB.amlodipine_rings(),
            "rano": SB.ranolazine_mpo(),
            "osmb": SB.hard_osimertinib(),
            "siga": SB.sitagliptin_replacement(),
            "zale": SB.zaleplon_with_other_formula(),
            "valt": SB.valsartan_smarts(),
            "med2": SB.median_tadalafil_sildenafil(),
            "dhop": SB.decoration_hop(),
            "shop": SB.scaffold_hop(),
            "fexo": SB.hard_fexofenadine(),
        }
        self.objective = objective
        if self.objective not in (
            list(self.guacamol_objs.keys()) + ["logP", None]
        ):
            raise NotImplementedError(
                f"Got unknown objective function {self.objective}"
            )

    def __call__(self, smile: Optional[str] = None) -> float:
        """
        Evaluates the objective function on a molecule with the SMILES
        representation.
        Input:
            smile: input molecule with the SMILES representation.
        Returns:
            The objective function evaluated on the input molecule.
        """
        if smile is None or len(smile) == 0 or self.objective is None:
            return None
        mol = Chem.MolFromSmiles(smile)
        if mol is None:
            return None
        if self.objective == "logP":
            logp = Crippen.MolLogP(mol)
            sa = sascorer.calculateScore(mol)
            cycle_length = self._cycle_score(mol)
            # Calculate the final penalized score. The magic numbers below
            # are the empirical means and standard deviations of the dataset.
            z_logp = (logp - 2.45777691) / 1.43341767
            z_sa = (sa - 3.05352042) / 0.83460587
            z_cycle_length = (cycle_length - 0.04861121) / 0.28746695
            penalized_logp = max(z_logp - z_sa - z_cycle_length, -float("inf"))
            return -1e12 if penalized_logp is None else penalized_logp
        score = self.guacamol_objs[self.objective].objective.score(smile)
        if score is None:
            return None
        return score if score >= 0 else None

    def _cycle_score(self, mol: Chem.Mol) -> int:
        """
        Calculates the cycle score for an input molecule.
        Input:
            mol: input molecule to calculate the cycle score for.
        Returns:
            The cycle score for the input molecule.
        """
        cycle_list = nx.cycle_basis(nx.Graph(rdmolops.GetAdjacencyMatrix(mol)))
        if len(cycle_list) == 0:
            cycle_length = 0
        else:
            cycle_length = max([len(j) for j in cycle_list])
        if cycle_length <= 6:
            cycle_length = 0
        else:
            cycle_length = cycle_length - 6
        return cycle_length


class WarfarinDosingOracle:
    """
    Warfarin weekly dose estimator using one of the linear regression models
    reported by IWPC. (NEJM 2009).
    """

    def __init__(
        self,
        transform: object,
        column_names: Union[Sequence[str], np.ndarray],
        mean_dose: Optional[float] = None,
        use_pharmacogenetic_algorithm: bool = True
    ):
        """
        Args:
            transform: a transform object to unnormalize input values.
            column_names: an optional list of the design dimension names.
            mean_dose: an optional warfarin dose to compare against.
            use_pharmacogenetic_algorithm: whether to use the pharmacogenetic
                dosing algorithm instead of the clinical dosing algorithm
                presented by IWPC. (NEJM 2009). Default True.
        """
        self.transform = transform
        self.column_names = (
            column_names.tolist()
            if isinstance(column_names, np.ndarray)
            else column_names
        )
        self.mean_dose = mean_dose
        self.eps = np.finfo(np.float32).eps
        self.dose = "Therapeutic Dose of Warfarin"
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

    def __call__(self, X: np.ndarray) -> np.ndarray:
        """
        Defines a quadratic cost function dependent on a patient's predicted
        dose and true stable warfarin dose.
        Input:
            X: the predicted weekly warfarin dose(s) and other attributes
                comprising the patient data.
        Returns:
            The cost associated with each patient's predicted dose.
        """
        X = X.copy()
        for col_name, transform in self.transform.transforms.items():
            idx = self.column_names.index(col_name)
            X[:, idx] = transform.inverse_transform(X[:, idx])
        gt_dose = self._dosing_algorithm(
            pd.DataFrame(X, columns=self.column_names), self.algorithm
        )
        y = np.square(X[:, self.column_names.index(self.dose)] - gt_dose)
        if self.mean_dose is None:
            return -1.0 * y.astype(X.dtype)
        ref_y = np.square(self.mean_dose - gt_dose)
        return ((ref_y - y) / (ref_y + self.eps)).astype(X.dtype)

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
        # Note that the decade age is divided by 10 in our input dataset, so
        # we multiply the coefficient for age by 10 here when compared to the
        # original algorithm to correct for this.
        return {
            "bias": 5.6044,
            "Age": -0.2614 * 10.0,
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
        # Note that the decade age is divided by 10 in our input dataset, so
        # we multiply the coefficient for age by 10 here when compared to the
        # original algorithm to correct for this.
        return {
            "bias": 4.0376,
            "Age": -0.2546 * 10.0,
            "Height (cm)": 0.0118,
            "Weight (kg)": 0.0134,
            "Race (OMB)_Asian": -0.6752,
            "Race (OMB)_Black or African American": 0.4060,
            "Race (OMB)_Unknown": 0.0443,
            self._enzyme_inducers: 1.2799,
            "Amiodarone (Cordarone)_1.0": -0.5695
        }
