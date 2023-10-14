"""
Main driver program for warfarin counterfactual generation learning.

Author(s):
    Michael Yao @michael-s-yao

Licensed under the MIT License. Copyright University of Pennsylvania 2023.
"""
import numpy as np
import pandas as pd
from collections.abc import Iterable
from itertools import product
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from tqdm import tqdm
from typing import Any, Dict, Sequence, Tuple, Union

from dataset import WarfarinDataset
from dosing import WarfarinDose


def dosage_cost(pred_dose: np.ndarray, gt_dose: np.ndarray) -> np.ndarray:
    """
    Defines a quadratic cost function dependent on a patient's predicted dose
    and true stable warfarin dose.
    Input:
        pred_dose: the predicted weekly warfarin dose(s) of the patient(s).
        gt_dose: the true stable weekly warfarin dose(s) of the patient(s).
    Returns:
        The cost associated with each patient's predicted dose.
    """
    return np.square(pred_dose - gt_dose)


def hyperparams(
    model: Union[DecisionTreeRegressor, RandomForestRegressor, Lasso]
) -> Tuple[Sequence[str], Iterable[Tuple[int]], int]:
    """
    Returns an iterable over the hyperparameter search space.
    Input:
        model: the model to search the hyperparameters over.
    Returns:
        hyperparam_names: the names of the hyperparameters.
        hyperparam_search: an iterable over the hyperparameter search space.
        total_searches: the cardinality of the hyperparameter search space.
    """
    if model == Lasso:
        alpha = [1.0, 2.0, 5.0]
        alpha = [
            [factor * x]
            for x in [1.0, 2.0, 5.0]
            for factor in [0.001, 0.01, 0.1, 1, 10, 100]
        ]
        return ["alpha"], alpha, len(alpha)
    max_depth = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, None]
    min_samples_split = [2, 4, 8, 16, 32, 64, 128]
    min_samples_leaf = [1, 2, 4, 8, 16]
    max_leaf_nodes = [2, 4, 8, 16, 32, 64, 128, None]
    hyperparams = [
        max_depth, min_samples_split, min_samples_leaf, max_leaf_nodes
    ]
    hyperparam_names = [
        "max_depth", "min_samples_split", "min_samples_leaf", "max_leaf_nodes"
    ]
    if model == RandomForestRegressor:
        n_estimators = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
        hyperparam_names.append("n_estimators")
        hyperparams.append(n_estimators)
    elif model != DecisionTreeRegressor:
        raise NotImplementedError(f"Unrecognized model type {model}")
    total_searches = np.prod([len(param_vals) for param_vals in hyperparams])
    return hyperparam_names, iter(product(*hyperparams)), total_searches


def search_hyperparams(
    model: Union[DecisionTreeRegressor, RandomForestRegressor, Lasso],
    savepath: Union[Path, str],
    cv: int = 10,
    seed: int = 42
) -> Tuple[Dict[str, Any], Sequence[float]]:
    """
    Performs a search over the hyperparameter search space.
    Input:
        model: the model to search the hyperparameters over.
        savepath: the CSV path to save the results of the search to.
        cv: number of folds for cross validation.
        seed: random seed. Default 42.
    Returns:
        best_hyperparams: a dictionary of the best hyperparameter values for
            the model.
        best_mses: the best model scores from cross validation using the best
            hyperparameters.
    """
    dataset = WarfarinDataset(seed=seed)
    dose = WarfarinDose()

    h, w, d = dataset.height, dataset.weight, dataset.dose
    X_train, pred_dose_train = dataset.train_dataset
    gt_dose_train = dose(X_train)
    cost_train = dosage_cost(pred_dose_train, gt_dose_train)

    if model == Lasso:
        scaler_h, scaler_w = StandardScaler(), StandardScaler()
        scaler_d = StandardScaler()
        for col, scaler in zip([h, w, d], [scaler_h, scaler_w, scaler_d]):
            X_train[col] = scaler.fit_transform(
                X_train[col].to_numpy()[:, None]
            )

    names, search, total = hyperparams(model)
    best_nmses, best_hyperparams = np.full(cv, -1e12), None
    random_forest_results = []
    for params in tqdm(search, desc="Hyperparameter Search", total=total):
        params = {name: val for name, val in zip(names, params)}
        if model == Lasso:
            params["max_iter"] = int(1e6)
            if "max_iter" not in names:
                names.append("max_iter")
        regressor = model(**params)
        nmses = cross_val_score(
            regressor,
            X_train,
            cost_train,
            cv=cv,
            scoring="neg_mean_squared_error"
        )
        if np.mean(nmses) > np.mean(best_nmses):
            best_nmses, best_hyperparams = nmses, params
        results = [val for _, val in params.items()]
        results += [np.mean(-1.0 * nmses), np.std(-1.0 * nmses, ddof=1)]
        random_forest_results.append(results)
    columns = names + ["cv_score_mean", "cv_score_std"]
    pd.DataFrame(random_forest_results, columns=columns).to_csv(savepath)
    return best_hyperparams, best_nmses


if __name__ == "__main__":
    model = RandomForestRegressor
    savepath = f"./{str(model).split('.')[-1][:-2]}_hyperparams.csv"
    search_hyperparams(model, savepath)
