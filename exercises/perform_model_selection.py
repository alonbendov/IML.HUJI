from __future__ import annotations
import numpy as np
import pandas as pd
import plotly
from sklearn import datasets
from IMLearn.metrics import mean_square_error
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
from IMLearn.learners.regressors import PolynomialFitting, LinearRegression, \
    RidgeRegression
from sklearn.linear_model import Lasso

from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots

X_MIN = -1.2
X_MAX = 2
TRAIN_PROPORTION = 2 / 3

MAX_DEGREE = 10
FOLDS = 5


def select_polynomial_degree(n_samples: int = 100, noise: float = 5):
    """
    Simulate data from a polynomial model and use cross-validation to select the best fitting degree

    Parameters
    ----------
    n_samples: int, default=100
        Number of samples to generate

    noise: float, default = 5
        Noise level to simulate in responses
    """
    # Question 1 - Generate dataset for model f(x)=(x+3)(x+2)(x+1)(x-1)(x-2) + eps for eps Gaussian noise
    x_values = np.random.uniform(X_MIN, X_MAX, n_samples)
    fx_values = (x_values + 3) * (x_values + 2) * (x_values + 1) * \
                (x_values - 1) * (x_values - 2) \
                + np.random.normal(0, noise, n_samples)

    # and split into training- and testing portions
    X_train, y_train, X_test, y_test = \
        split_train_test(pd.DataFrame(x_values), pd.Series(fx_values),
                         TRAIN_PROPORTION)

    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10
    degrees = np.arange(MAX_DEGREE + 1)
    train_errors = np.zeros(degrees.shape[0])
    validation_errors = np.zeros(degrees.shape[0])

    for deg in degrees:
        estimator = PolynomialFitting(deg)
        train_errors[deg], validation_errors[deg] = cross_validate(estimator,
                                                                   X_train.to_numpy(),
                                                                   y_train.to_numpy(),
                                                                   mean_square_error,                                                                   FOLDS)
    best_deg = np.argmin(validation_errors)
    best_err = np.round(validation_errors[best_deg], 2)

    # Plotting
    fig = go.Figure()
    fig.add_trace(go.Scatter(mode='lines', x=degrees, y=train_errors,
                             name="Train Error"))
    fig.add_trace(
        go.Scatter(mode='lines', x=degrees, y=validation_errors,
                   name="Validation Error"))
    fig.update_layout(title=f"{FOLDS}-Folds CV Results for Polynomial Fitting"
                            f"<br><sup>"
                            f"M={n_samples}, noise={noise}"
                            f" | Best validation error of {best_err} "
                            f"achieved on deg={best_deg}</sup>",
                      xaxis={'title': 'Degree of the Polynomial Model'},
                      yaxis={'title': 'Average MSE'})
    fig.show()
    fig.write_image(f'.\\ex5_graphs\\poly-deg-eval-noise={noise}.jpeg',
                    scale=2)

    # Question 3 - Using best value of k, fit a k-degree polynomial model and report test error
    model = PolynomialFitting(int(best_deg))
    model.fit(X_train.to_numpy(), y_train.to_numpy())
    error = np.round(
        mean_square_error(y_test.to_numpy(), model.predict(X_test.to_numpy())),
        2)

    print(f"Report of m={n_samples} and noise={noise}:")
    print(f"Best k-folds validation error of {best_err} on deg={best_deg}")
    print(f"When training a model of deg={best_deg} on all of the train data"
          f" error of {error} was achieved.")
    print()


MIN_LAMBDA = 0
MAX_LAMBDA = 10


def select_regularization_parameter(n_samples: int = 50,
                                    n_evaluations: int = 500):
    """
    Using sklearn's diabetes dataset use cross-validation to select the best fitting regularization parameter
    values for Ridge and Lasso regressions

    Parameters
    ----------
    n_samples: int, default=50
        Number of samples to generate

    n_evaluations: int, default = 500
        Number of regularization parameter values to evaluate for each of the algorithms
    """
    # Question 6 - Load diabetes dataset and split into training and testing portions

    X, y = datasets.load_diabetes(return_X_y=True, as_frame=True)
    X_train, y_train, X_test, y_test = split_train_test(X, y, n_samples /
                                                        X.shape[0])
    X_train, y_train, X_test, y_test = X_train.to_numpy(), y_train.to_numpy(), X_test.to_numpy(), y_test.to_numpy()
    # X, y = datasets.load_diabetes(return_X_y=True)
    # X_train, y_train, X_test, y_test = X[:n_samples], y[:n_samples], X[n_samples:], y[n_samples:]
    assert (X_train.shape[0] == n_samples)

    # Question 7 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions
    lambdas = np.linspace(MIN_LAMBDA, MAX_LAMBDA, n_evaluations)
    lasso_err_train, ridge_err_train, lasso_err_val, ridge_err_val = \
        np.zeros(n_evaluations), np.zeros(n_evaluations), \
        np.zeros(n_evaluations), np.zeros(n_evaluations)
    for i, lam in enumerate(lambdas):
        lasso = Lasso(lam, max_iter=1000000)
        ridge = RidgeRegression(lam)

        lasso_err_train[i], lasso_err_val[i] = \
            cross_validate(lasso, X_train, y_train,
                           mean_square_error)
        ridge_err_train[i], ridge_err_val[i], = \
            cross_validate(ridge, X_train, y_train,
                           mean_square_error)

    fig = make_subplots(2, 1, subplot_titles=('Lasso', 'Ridge'), shared_xaxes=True)
    fig.update_layout(title="Train vs Validation Performance with Change of the Regularization"
                            "<br>"
                            "<sup>Measured on 5-Fold CV</sup>")

    fig.update_xaxes(showgrid=False, row=1, col=1)
    fig.update_xaxes(title_text="Lambda (Regularization Factor)", row=2, col=1)
    fig.update_yaxes(title_text="MSE", row=1, col=1)
    fig.update_yaxes(title_text="MSE", row=2, col=1)

    fig.add_trace(
        go.Scatter(mode='lines', x=lambdas, y=lasso_err_train,
                   name="Train Error"), row=1, col=1)
    fig.add_trace(
        go.Scatter(mode='lines', x=lambdas, y=lasso_err_val,
                   name="Validation Error"), row=1, col=1)
    fig.add_trace(
        go.Scatter(mode='lines', x=lambdas, y=ridge_err_train,
                   name="Train Error"), row=2, col=1)
    fig.add_trace(
        go.Scatter(mode='lines', x=lambdas, y=ridge_err_val,
                   name="Validation Error"), row=2, col=1)

    fig.show()
    fig.write_image(f'.\\ex5_graphs\\lasso-vs-ridge.jpeg',
                    scale=2)

    # Question 8 - Compare best Ridge model, best Lasso model and Least Squares model
    lasso_best_ind = np.argmin(lasso_err_val)
    ridge_best_ind = np.argmin(ridge_err_val)

    lasso_err_val_best = lasso_err_val[lasso_best_ind]
    ridge_err_val_best = ridge_err_val[ridge_best_ind]

    lasso_best_lam = lambdas[lasso_best_ind]
    ridge_best_lam = lambdas[ridge_best_ind]

    lasso_best = Lasso(lasso_best_lam)
    ridge_best = RidgeRegression(ridge_best_lam)

    lasso_best.fit(X_train, y_train)
    ridge_best.fit(X_train, y_train)

    lasso_best_err = mean_square_error(y_test, lasso_best.predict(X_test))
    ridge_best_err = mean_square_error(y_test, ridge_best.predict(X_test))

    print(f"Report for Ridge:")
    print(f"Best k-folds validation error of {lasso_err_val_best} on lambda={lasso_best_lam}")
    print(f"When training a model of lambda={lasso_best_lam} on all of the train "
          f"data error of {lasso_best_err} was achieved.")
    print()

    print(f"Report for Ridge:")
    print(f"Best k-folds validation error of {ridge_err_val_best} on lambda={ridge_best_lam}")
    print(f"When training a model of lambda={ridge_best_lam} on all of the train "
          f"data error of {ridge_best_err} was achieved.")
    print()

if __name__ == '__main__':
    np.random.seed(0)
    select_polynomial_degree()
    select_polynomial_degree(noise=0)
    select_polynomial_degree(1500, 10)
    select_regularization_parameter()
