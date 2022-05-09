import numpy as np

from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
from typing import Tuple
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from math import atan2, pi


def load_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset for comparing the Gaussian Naive Bayes and LDA classifiers. File is assumed to be an
    ndarray of shape (n_samples, 3) where the first 2 columns represent features and the third column the class

    Parameters
    ----------
    filename: str
        Path to .npy data file

    Returns
    -------
    X: ndarray of shape (n_samples, 2)
        Design matrix to be used

    y: ndarray of shape (n_samples,)
        Class vector specifying for each sample its class

    """
    data = np.load(filename)
    return data[:, :2], data[:, 2].astype(int)


def run_perceptron():
    """
    Fit and plot fit progression of the Perceptron algorithm over both the linearly separable and inseparable datasets

    Create a line plot that shows the perceptron algorithm's training loss values (y-axis)
    as a function of the training iterations (x-axis).
    """
    for n, f in [
        ("Linearly Separable", "..\\datasets\\linearly_separable.npy"),
        ("Linearly Inseparable", "..\\datasets\\linearly_inseparable.npy")]:
        # Load dataset
        X, y = load_dataset(f)

        # Fit Perceptron and record loss in each fit iteration
        losses = []
        add_loss = lambda fit, _, __: losses.append(fit.loss(X, y))
        p = Perceptron(callback=add_loss)
        p.fitted_ = True
        p.fit(X, y)

        # Plot figure of loss as function of fitting iteration
        fig = px.line(losses)
        fig.update_layout(
            xaxis={'title': 'times the coefficient have been changed '},
            yaxis={'title': 'loss'},
            title=f'Perceptron on {n} data | Change in loss while fitting',
            showlegend=False
        )
        fig.write_image(f'.\\ex3_graphs\\{n} losses.jpeg', scale=2)
        fig.show()


def get_ellipse(mu: np.ndarray, cov: np.ndarray):
    """
    Draw an ellipse centered at given location and according to specified covariance matrix

    Parameters
    ----------
    mu : ndarray of shape (2,)
        Center of ellipse

    cov: ndarray of shape (2,2)
        Covariance of Gaussian

    Returns
    -------
        scatter: A plotly trace object of the ellipse
    """
    l1, l2 = tuple(np.linalg.eigvalsh(cov)[::-1])
    theta = atan2(l1 - cov[0, 0], cov[0, 1]) if cov[0, 1] != 0 else (
        np.pi / 2 if cov[0, 0] < cov[1, 1] else 0)
    t = np.linspace(0, 2 * pi, 100)
    xs = (l1 * np.cos(theta) * np.cos(t)) - (l2 * np.sin(theta) * np.sin(t))
    ys = (l1 * np.sin(theta) * np.cos(t)) + (l2 * np.cos(theta) * np.sin(t))

    return go.Scatter(x=mu[0] + xs, y=mu[1] + ys, mode="lines",
                      marker_color="black")


def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """
    for f in ["..\\datasets\\gaussian1.npy", "..\\datasets\\gaussian2.npy"]:
        # Load dataset
        dataset_name = f[f.rfind('\\') + 1:f.rfind('.')]
        ds = load_dataset(f)
        X, y = ds[0], ds[1]
        n_classes = len(np.unique(y))

        # Fit models and predict over training set
        lda = LDA()
        naive = GaussianNaiveBayes()
        lda.fit(X, y)
        naive.fit(X, y)
        pred_lda = lda.predict(X)
        pred_naive = naive.predict(X)

        # Plot a figure with two suplots, showing the Gaussian Naive Bayes predictions on the left and LDA predictions
        # on the right. Plot title should specify dataset used and subplot titles should specify algorithm and accuracy
        # Create subplots
        from IMLearn.metrics import accuracy
        fig = make_subplots(1, 2, subplot_titles=(
            f'GNBC, Accuracy={accuracy(y, pred_naive)}',
            f'LDA, Accuracy={accuracy(y, pred_lda)}'))
        fig.update_layout(showlegend=False,
                          title=f'Learning Data from {dataset_name}')

        # Add traces for data-points setting symbols and colors
        trace_naive = go.Scatter(x=X[:, 0], y=X[:, 1], mode='markers',
                                 marker={'color': pred_naive, 'symbol': y})
        trace_lda = go.Scatter(x=X[:, 0], y=X[:, 1], mode='markers',
                               marker={'color': pred_lda, 'symbol': y})

        # Add `X` dots specifying fitted Gaussians' means
        fig.add_trace(go.Scatter(
            x=naive.mu_[:, 0],
            y=naive.mu_[:, 1],
            mode="markers", marker=dict(color="green", symbol='hexagram', size=9)),
            row=1, col=1
        )
        fig.add_trace(go.Scatter(
            x=lda.mu_[:, 0],
            y=lda.mu_[:, 1],
            mode="markers", marker=dict(color="green", symbol='hexagram', size=9)),
            row=1, col=2
        )

        # Add ellipses depicting the covariances of the fitted Gaussians
        for i in range(n_classes):
            naive_elipse = get_ellipse(naive.mu_[i], np.diag(naive.vars_[i]))
            fig.add_trace(naive_elipse, 1, 1)

            lda_elipse = get_ellipse(lda.mu_[i], lda.cov_)
            fig.add_trace(lda_elipse, 1, 2)

        fig.add_trace(trace_naive, 1, 1)
        fig.add_trace(trace_lda, 1, 2)
        fig.show()

        fig.write_image(f'.\\ex3_graphs\\{dataset_name}.jpeg', scale=1)


if __name__ == '__main__':
    np.random.seed(0)
    run_perceptron()
    compare_gaussian_classifiers()
