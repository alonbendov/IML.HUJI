from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from tqdm import tqdm

pio.templates.default = "simple_white"
import plotly.express as px
import pandas as pd


def test_univariate_gaussian():
    loc, var = 10, 1
    np.random.seed(0)
    random = np.random.normal(loc, var, 1000)

    # Question 1 - Draw samples and print fitted model
    model = UnivariateGaussian(True)
    model.fit(random[0:101])
    print(f'({model.mu_}, {model.var_})')

    # Question 2 - Empirically showing sample mean is consistent
    iterations = 100
    abs_dist = np.zeros(iterations)
    samples = np.array(range(10, 1010, 10))
    for i, count in enumerate(samples):
        model = UnivariateGaussian(True)
        model.fit(random[0:count + 1])
        abs_dist[i] = abs(loc - model.mu_)

    df = pd.DataFrame(dict(x=samples, y=abs_dist))
    fig = px.line(df, x='x', y='y',
                  title="Convergences to RealExpectation with the Increase of Sample Size",
                  labels=dict(x="Sample size",
                              y="| EstimatedExpectation - RealExpectation |"))
    fig.show()

    # Question 3 - Plotting Empirical PDF of fitted model
    model.fit(random)
    pdf_x = np.linspace(0, 20, 1000)
    pdf_y = model.pdf(pdf_x)
    df = pd.DataFrame(dict(x=pdf_x, y=pdf_y))
    fig = px.scatter(df, x='x', y='y',
                     title="PDF Calculated from the Random Sample",
                     labels=dict(x='X', y='PDF(X)'))
    fig.show()


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    real_mu = np.array([0, 0, 4, 0])
    real_cov = np.array(
        [[1, 0.2, 0, 0.5], [0.2, 2, 0, 0], [0, 0, 1, 0], [0.5, 0, 0, 1]])
    random = np.random.multivariate_normal(real_mu, real_cov, 1000)
    model = MultivariateGaussian()
    model.fit(random)

    print(model.mu_)
    print(model.cov_)

    # Question 5 - Likelihood evaluation
    size = 200
    linsp = np.linspace(-10, 10, size)
    map = np.zeros((size, size))

    likelihood_mu = np.zeros(4)
    arg_max = [0, 0]
    max_log_likelihood = MultivariateGaussian.log_likelihood(
        likelihood_mu, real_cov, random)

    for i, f1 in tqdm(enumerate(linsp)):
        for j, f3 in enumerate(linsp):
            likelihood_mu[0] = f1
            likelihood_mu[2] = f3
            map[i, j] = MultivariateGaussian.log_likelihood(
                likelihood_mu, real_cov, random)

            if map[i, j] > max_log_likelihood:
                max_log_likelihood = map[i, j]
                arg_max = [f1, f3]

    print(map)
    fig = px.imshow(map, x=linsp, y=linsp, labels=dict(x="f3", y="f1"),
                    title="Log-liklelihood of the randomized variables with "
                          "the original covariance and mean=[f1,0,f3,0]")
    fig.show()

    # Question 6 - Maximum likelihood
    print(
        f"Maximum log-likelihood achieved on the coordinates ("
        f"{round(arg_max[0], 4)},{round(arg_max[1], 4)})")


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
