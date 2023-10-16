"""
Builds a warfarin dosage-associated cost estimator.

Author(s):
    Michael Yao @michael-s-yao

Licensed under the MIT License. Copyright University of Pennsylvania 2023.
"""
import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
from collections.abc import Iterable
from itertools import product
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Lasso
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from tqdm import tqdm
from typing import Any, Dict, Optional, Sequence, Tuple, Union

from cost import dosage_cost
from dataset import WarfarinDataset
from dosing import WarfarinDose


def hyperparams(
    model: Any, seed: int = 42
) -> Tuple[Sequence[str], Iterable[Tuple[int]], int]:
    """
    Returns an iterable over the hyperparameter search space.
    Input:
        model: the model to search the hyperparameters over.
        seed: random seed. Default 42.
    Returns:
        hyperparam_names: the names of the hyperparameters.
        hyperparam_search: an iterable over the hyperparameter search space.
        total_searches: the cardinality of the hyperparameter search space.
    """
    if model == MLPRegressor:
        hidden_layer_sizes = [(32,), (64,), (32,) * 8, (128,) * 2]
        batch_size = [16, 32, 64]
        max_iter = [500]
        learning_rate_init = [0.001, 0.01, 0.1]
        early_stopping = [True]
        random_state = [seed]
        hyperparam_names = [
            "hidden_layer_sizes",
            "batch_size",
            "max_iter",
            "learning_rate_init",
            "early_stopping",
            "random_state"
        ]
        hyperparams = [
            hidden_layer_sizes,
            batch_size,
            max_iter,
            learning_rate_init,
            early_stopping,
            random_state
        ]
        total_searches = np.prod([len(vals) for vals in hyperparams])
        return hyperparam_names, iter(product(*hyperparams)), total_searches
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
    elif model == GradientBoostingRegressor:
        max_depth = [1, 2, 4, 8, 16, 32]
        n_estimators = [128, 256, 512, 1024]
        learning_rate = [0.001, 0.01, 0.1, 1]
        hyperparams = [
            max_depth,
            min_samples_split,
            min_samples_leaf,
            max_leaf_nodes,
            n_estimators,
            learning_rate
        ]
        hyperparam_names += ["n_estimators", "learning_rate"]
    elif model != DecisionTreeRegressor:
        raise NotImplementedError(f"Unrecognized model type {model}")
    total_searches = np.prod([len(param_vals) for param_vals in hyperparams])
    return hyperparam_names, iter(product(*hyperparams)), total_searches


def search_hyperparams(
    model: Any,
    savepath: Union[Path, str],
    p: int = 1,
    cv: int = 10,
    seed: int = 42
) -> Tuple[Dict[str, Any], Sequence[float]]:
    """
    Performs a search over the hyperparameter search space.
    Input:
        model: the model to search the hyperparameters over.
        savepath: the CSV path to save the results of the search to.
        p: power of cost function to fit to. Default 1.
        cv: number of folds for cross validation. Default 10.
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
    cost_train_p = np.power(cost_train, p)
    scaler_cost = StandardScaler()
    cost_train = scaler_cost.fit_transform(cost_train_p.to_numpy()[:, None])

    if model == Lasso or model == MLPRegressor:
        scaler_h, scaler_w = StandardScaler(), StandardScaler()
        scaler_d = StandardScaler()
        for col, scaler in zip([h, w, d], [scaler_h, scaler_w, scaler_d]):
            X_train[col] = scaler.fit_transform(
                X_train[col].to_numpy()[:, None]
            )
    if model == GradientBoostingRegressor or model == MLPRegressor:
        cost_train = np.squeeze(cost_train)

    names, search, total = hyperparams(model)
    best_nmses, best_hyperparams = np.full(cv, -1e12), None
    search_results = []
    for params in tqdm(
        search, desc="Hyperparameter Search", total=total, leave=False
    ):
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
        search_results.append(results)
    columns = names + ["cv_score_mean", "cv_score_std"]
    pd.DataFrame(search_results, columns=columns).to_csv(savepath)

    return best_hyperparams, best_nmses


def plot_error(
    train: Tuple[np.ndarray],
    test: Tuple[np.ndarray],
    savepath: Optional[Union[Path, str]] = None
) -> None:
    plt.figure(figsize=(10, 4))
    for i, ((y_pred, y), label) in enumerate(
        zip([train, test], ["Training", "Test"])
    ):
        plt.hist(
            y_pred - y,
            density=True,
            bins=np.linspace(-20, 20, 100),
            label=label,
            alpha=(0.6 - (0.1 * i))
        )
    plt.xlabel("Predicted Cost - True Cost")
    plt.ylabel("Density")
    plt.legend()
    if savepath is None:
        plt.show()
    else:
        plt.savefig(savepath, dpi=600, transparent=True, bbox_inches="tight")
    plt.close()


def build_cost_func(
    model: Any,
    hparams: Union[Path, str] = "./hparams.json",
    seed: int = 42,
    plotpath: Optional[Union[Path, str]] = None,
    savepath: Optional[Union[Path, str]] = None
) -> float:
    """
    Trains and tests a specified model type as a warfarin cost estimator.
    Input:
        model: the model to search the hyperparameters over.
        hparams: the file path to the JSON file with the model hyperparameters.
        seed: random seed. Default 42.
        plotpath: optional path to save the histogram plot to. Default None.
        savepath: optional path to save the model to. Default None.
    Returns:
        RMSE value on the test dataset.
    """
    dataset = WarfarinDataset(seed=seed)
    dose = WarfarinDose()

    with open(hparams, "rb") as f:
        hparams = json.load(f)[str(model).split(".")[-1][:-2]]
    p = hparams.pop("p")

    h, w, d = dataset.height, dataset.weight, dataset.dose
    X_train, pred_dose_train = dataset.train_dataset
    gt_dose_train = dose(X_train)
    X_test, pred_dose_test = dataset.test_dataset
    cost_train = dosage_cost(pred_dose_train, gt_dose_train)
    gt_dose_test = dose(X_test)
    cost_train_p = np.power(cost_train, p)
    scaler_cost = StandardScaler()
    cost_train = scaler_cost.fit_transform(cost_train_p.to_numpy()[:, None])
    cost_test = dosage_cost(pred_dose_test, gt_dose_test)

    if model == Lasso or model == MLPRegressor:
        scaler_h, scaler_w = StandardScaler(), StandardScaler()
        scaler_d = StandardScaler()
        for col, scaler in zip([h, w, d], [scaler_h, scaler_w, scaler_d]):
            X_train[col] = scaler.fit_transform(
                X_train[col].to_numpy()[:, None]
            )
            X_test[col] = scaler.transform(X_test[col].to_numpy()[:, None])
    if model == GradientBoostingRegressor or model == MLPRegressor:
        cost_train = np.squeeze(cost_train)

    regressor = model(**hparams)
    regressor.fit(X_train, cost_train)

    pred_cost_test = np.power(
        np.squeeze(
            scaler_cost.inverse_transform(regressor.predict(X_test)[:, None])
        ),
        1.0 / p
    )
    pred_cost_train = np.power(
        np.squeeze(
            scaler_cost.inverse_transform(regressor.predict(X_train)[:, None])
        ),
        1.0 / p
    )
    cost_train = np.squeeze(cost_train)[:, None]
    cost_train = np.power(
        np.squeeze(scaler_cost.inverse_transform(cost_train)), 1.0 / p
    )
    plot_error(
        (pred_cost_train, np.squeeze(cost_train)),
        (pred_cost_test, cost_test),
        plotpath
    )
    if savepath is not None:
        with open(savepath, "wb") as f:
            pickle.dump(regressor, f)
    return np.sqrt(np.mean(np.square(pred_cost_test - cost_test)))


def select_model() -> Any:
    """
    Selects a model architecture to train as the cost estimator.
    Input:
        None.
    Returns:
        Model to train.
    """
    models = {
        "Lasso": Lasso,
        "DecisionTreeRegressor": DecisionTreeRegressor,
        "GradientBoostingRegressor": GradientBoostingRegressor,
        "RandomForestRegressor": RandomForestRegressor,
        "MLPRegressor": MLPRegressor
    }
    parser = argparse.ArgumentParser("Warfarin dosage cost estimator.")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=list(models.keys()),
        help="Predictive cost model backbone."
    )
    return models[parser.parse_args().model]


if __name__ == "__main__":
    build_cost_func(select_model())
