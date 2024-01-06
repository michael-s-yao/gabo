"""
Defines the oracle objectives for molecule property optimization tasks.

Author(s):
    Michael Yao

Citation(s):
    [1] Brown N, Fiscato M, Segler MHS, Vaucher AC. GuacaMol: Benchmarking
        models for de novo molecular design. J Chem Inf Model 59(3):1096-08.
        (2019). https://doi.org/10.1021/acs.jcim.8b00839

Licensed under the MIT License. Copyright University of Pennsylvania 2023.
"""
import networkx as nx
import models.oracle.sascorer as sascorer
from rdkit import Chem
from rdkit.Chem import Crippen
from rdkit.Chem import rdmolops
from guacamol import standard_benchmarks as SB
from typing import Optional


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
