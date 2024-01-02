import torch
import torch.nn as nn
import math
import selfies as sf
from torch.utils.data import TensorDataset, DataLoader
from typing import Any, Dict, Sequence, Optional

import networkx as nx
import sascorer
from rdkit import Chem
from rdkit.Chem import Crippen
from rdkit.Chem import rdmolops
from guacamol import standard_benchmarks as SB


class MoleculeObjective:
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


def finetune(
    train_x,
    train_y_scores,
    objective,
    model: nn.Module,
    mll,
    lr: float,
    num_update_epochs: int,
    clip_grad_norm: Optional[float] = 1.0
):
    """
    Finetune VAE end to end with the surrogate model.
    Input:
        TODO
    Returns:
        TODO
    """
    objective.vae.train()
    model.train()
    optimizer = torch.optim.Adam(
        [
            {"params": objective.vae.parameters()},
            {"params": model.parameters(), "lr": lr}
        ],
        lr=lr
    )

    max_string_length = len(max(train_x, key=len))
    batch_size = max(1, int(2560 / max_string_length))
    num_batches = math.ceil(len(train_x) / batch_size)
    for _ in range(num_update_epochs):
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            stop_idx = (batch_idx + 1) * batch_size
            batch = train_x[start_idx:stop_idx]
            z, vae_loss = objective.vae_forward(batch)
            batch_y = train_y_scores[start_idx:stop_idx]
            batch_y = torch.tensor(batch_y).float()
            pred = model(z)

            surr_loss = -mll(pred, batch_y.cuda())
            loss = vae_loss + surr_loss

            optimizer.zero_grad()
            loss.backward()
            if clip_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    objective.vae.parameters(), max_norm=clip_grad_norm
                )
            optimizer.step()
    objective.vae.eval()
    model.eval()

    return objective, model


def update_surr_model(
    model,
    mll,
    train_z,
    train_y,
    lr: float,
    num_epochs: int,
    device: torch.device = torch.device("cpu")
):
    model = model.train()

    optimizer = torch.optim.Adam(
        [{"params": model.parameters(), "lr": lr}], lr=lr
    )
    train_dataset = TensorDataset(train_z.to(device), train_y.to(device))
    train_loader = DataLoader(
        train_dataset, batch_size=min(len(train_y), 128), shuffle=True
    )

    for _ in range(num_epochs):
        for (inputs, scores) in train_loader:
            optimizer.zero_grad()
            output = model(inputs.cuda())
            loss = -mll(output, scores.cuda())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

    model = model.eval()
    return model


def smiles_to_tokens(
    smiles: Sequence[str], vocab: Dict[str, Any]
) -> torch.Tensor:
    """
    Converts a list of SMILES molecule representations as a tensor of tokens.
    Input:
        smiles: a list of SMILES molecule representations.
    Returns:
        The molecules represented as a tensor of tokens.
    """
    start, stop = None, None
    for tok in vocab.keys():
        if "start" in tok:
            start = tok
        elif "stop" in tok:
            stop = tok
    tokens = []
    for smi in smiles:
        seq = [vocab[start]]
        seq += [vocab[tok] for tok in sf.split_selfies(sf.encoder(smi))]
        tokens.append(seq + [vocab[stop]])
    max_len = len(max(tokens, key=len))
    return torch.tensor([
        seq + ([vocab[stop]] * (max_len - len(seq))) for seq in tokens
    ])
