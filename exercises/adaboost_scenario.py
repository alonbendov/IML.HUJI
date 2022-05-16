import numpy as np
from typing import Tuple
from IMLearn.metalearners.adaboost import AdaBoost
from IMLearn.learners.classifiers import DecisionStump
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size

    Parameters
    ----------
    n: int
        Number of samples to generate

    noise_ratio: float
        Ratio of labels to invert

    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples

    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000,
                              test_size=500):
    (train_X, train_y), (test_X, test_y) = generate_data(train_size,
                                                         noise), generate_data(
        test_size, noise)
    if (noise > 0):
        ftag = f"N{noise}_"
        ttag = f"Noise = {noise}: "
    else:
        ftag = ""
        ttag = ""

    # Question 1: Train- and test errors of AdaBoost in noiseless case
    model = AdaBoost(DecisionStump, n_learners)
    model.fit(train_X, train_y)
    learner_amounts = np.arange(n_learners) + 1
    train_errs = np.vectorize(
        lambda x: model.partial_loss(train_X, train_y, x))(learner_amounts)
    test_errs = np.vectorize(lambda x: model.partial_loss(test_X, test_y, x))(
        learner_amounts)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=learner_amounts, y=train_errs, name='Train Error'))
    fig.add_trace(
        go.Scatter(x=learner_amounts, y=test_errs, name='Test Error'))

    fig.update_layout(title=f'{ttag}Errors on Train and Test Samples '
                            'as Function of AdaBoost Iterations',
                      xaxis_title='Number of fitted learners',
                      yaxis_title='Error')

    fig.write_image(f'.\\ex4_graphs\\{ftag}train_test_losses.jpeg', scale=2)
    fig.show()

    # Question 2: Plotting decision surfaces
    T = [5, 50, 100, 250]
    lims = np.array([np.r_[train_X, test_X].min(axis=0),
                     np.r_[train_X, test_X].max(axis=0)]).T + np.array(
        [-.1, .1])
    for t in T:
        fig = go.Figure()
        contour = decision_surface(lambda x: model.partial_predict(x, t),
                                   lims[0], lims[1])
        scatter = go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode='markers',
                             marker=dict(color=test_y, colorscale=custom,
                                         line=dict(width=0.5, color='black')))
        fig.add_trace(scatter)
        fig.add_trace(contour)
        title = f"{ttag}Test Set with Desicion Boundary of Ensemble Size = " + str(
            t)
        fig.update_layout(title=title, xaxis=dict(title="x"),
                          yaxis=dict(title="y"))
        fig.write_image(
            f'.\\ex4_graphs\\{ftag}boundaries_with_n_models_{t}.jpeg', scale=2)
        fig.show()

    # Question 3: Decision surface of best performing ensemble
    best_T_ind = np.argmin(test_errs)
    best_T = best_T_ind + 1
    best_T_err = test_errs[best_T_ind]

    # Generating Title
    best_T_other = np.where(test_errs == best_T_err)[0] + 1
    others_str = ' '.join([val + "<br>" if i % 20 == 0 else val for i, val in
                           enumerate(best_T_other.astype('str'))])
    title = f"{ttag}Best Ensemble Size Was {best_T}<br><sup>This" \
            f" was the best performer with error of {best_T_err} " \
            f"on the test set.<br>This result was achieved also for sizes: " \
            f" {others_str}</sup>"

    fig = go.Figure()
    contour = decision_surface(lambda x: model.partial_predict(x, best_T - 1),
                               lims[0],
                               lims[1])
    scatter = go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode='markers',
                         marker=dict(color=test_y, colorscale=custom,
                                     line=dict(width=0.5, color='black')))
    fig.add_trace(scatter)
    fig.add_trace(contour)
    fig.update_layout(title=dict(
        text=title,
    ), xaxis=dict(title="x"),
        yaxis=dict(title="y"))
    fig.write_image(f'.\\ex4_graphs\\{ftag}best_performer.jpeg',
                    scale=2)
    fig.show()

    # Question 4: Decision surface with weighted samples
    fig = go.Figure()
    contour = decision_surface(lambda x: model.partial_predict(x, T[-1]),
                               lims[0],
                               lims[1])
    scatter = go.Scatter(x=train_X[:, 0], y=train_X[:, 1], mode='markers',
                         marker=dict(color=train_y, colorscale=custom,
                                     line=dict(width=0.5, color='black'),
                                     size=model.D_ / np.max(model.D_) * 5))
    fig.add_trace(scatter)
    fig.add_trace(contour)
    title = f"{ttag}Train Set with Decision Boundary of Ensemble Size = {str(T[-1])}" \
            f"<br><sup>Point sizes are proportional to weight (in D)</sup>"
    fig.update_layout(title=title, xaxis=dict(title="x"),
                      yaxis=dict(title="y"))
    fig.write_image(f'.\\ex4_graphs\\{ftag}D_test_dots.jpeg',
                    scale=2)
    fig.show()


if __name__ == '__main__':
    np.random.seed(0)
    fit_and_evaluate_adaboost(0)
    fit_and_evaluate_adaboost(0.4)
