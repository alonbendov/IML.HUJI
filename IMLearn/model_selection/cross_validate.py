from __future__ import annotations
from copy import deepcopy
from typing import Tuple, Callable
import numpy as np
from IMLearn import BaseEstimator


def cross_validate(estimator: BaseEstimator, X: np.ndarray, y: np.ndarray,
                   scoring: Callable[[np.ndarray, np.ndarray, ...], float],
                   cv: int = 5) -> Tuple[float, float]:
    """
    Evaluate metric by cross-validation for given estimator

    Parameters
    ----------
    estimator: BaseEstimator
        Initialized estimator to use for fitting the data

    X: ndarray of shape (n_samples, n_features)
       Input data to fit

    y: ndarray of shape (n_samples, )
       Responses of input data to fit to

    scoring: Callable[[np.ndarray, np.ndarray, ...], float]
        Callable to use for evaluating the performance of the cross-validated model.
        When called, the scoring function receives the true- and predicted values for each sample
        and potentially additional arguments. The function returns the score for given input.

    cv: int
        Specify the number of folds.

    Returns
    -------
    train_score: float
        Average train score over folds

    validation_score: float
        Average validation score over folds
    """
    n_samples, n_features = X.shape
    train_scores = []
    validation_scores = []


    indices_permutation = np.arange(n_samples)

    indices_folds = np.array_split(indices_permutation, cv)

    for i in range(cv):
        val_idx = np.array(indices_folds[i])
        train_idx = np.concatenate(
            [indices_folds[j] for j in range(cv) if j != i])

        estimator.fit(X[train_idx], y[train_idx])

        train_pred = estimator.predict(X[train_idx])
        val_pred = estimator.predict(X[val_idx])

        train_score = scoring(y[train_idx], train_pred)
        val_score = scoring(y[val_idx], val_pred)

        train_scores.append(train_score)
        validation_scores.append(val_score)

    return np.average(train_scores), np.average(validation_scores)
