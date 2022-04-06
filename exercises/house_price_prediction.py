import os
import sys

import plotly.graph_objects
from numpy.random import sample

from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

pio.templates.default = "simple_white"
DATA_FILE = "..\\datasets\\house_prices.csv"


def load_data(filename: str):
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    df = pd.read_csv(filename)
    # parse dates
    df['date'] = pd.to_datetime(df['date'], format='%Y%m%dT000000',
                                errors='coerce')
    # drop invalid values
    # print(df.shape)
    df = df.dropna()
    # print(df.shape)
    df = df[df['price'] > 0]
    df = df[df['sqft_lot15'] >= 0]
    # print(df.shape)
    # one-hot encoding for zipcodes
    dummies = pd.get_dummies(df['zipcode'])
    dummies = dummies.rename(columns=lambda x: 'zip' + str(int(x)))
    df = pd.concat([df.drop(['zipcode'], axis=1), dummies], axis=1)
    # dates -> features
    dates: pd.Series = df['date']
    first_sell = dates.min()
    from_first = (dates - first_sell).dt.days
    day_of_year = dates.dt.dayofyear
    age_at_sell = dates.dt.year - df['yr_built']
    from_first = from_first.rename("days_delta")
    day_of_year = day_of_year.rename("day_of_year")
    age_at_sell = age_at_sell.rename('age_at_sell')
    df = pd.concat([df.drop(['date'], axis=1), from_first, day_of_year,
                    age_at_sell],axis=1)
    # lat lng removal
    df = df.drop(['lat', 'long'], axis=1)
    # id removal
    df = df.drop(['id'], axis=1)
    X = df.drop(['price'], axis=1)
    y = df['price']
    return X, y


def feature_evaluation(X: pd.DataFrame, y: pd.Series,
                       output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    if output_path[-1] != '\\' and output_path[-1] != '/':
        output_path += '\\'

    normalized_y = y / y.max()
    for colname in X.columns:
        col = X[colname]
        pearson = np.cov(col, y)[0][1] / (np.std(col) * np.std(y))
        normalized_col = col / col.max()

        fig = px.scatter(x=normalized_col, y=normalized_y)
        fig.update_layout(
            xaxis_title=f'normalied feature "{colname}"',
            yaxis_title='normalized price',
            title=f'{colname} vs price\nρ={pearson}.')
        fig.write_image(output_path + colname + ".jpeg")


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    X, y = load_data(DATA_FILE)

    # Question 2 - Feature evaluation with respect to response
    feature_evaluation(X, y,".\\ex2_graphs\\")

    # Question 3 - Split samples into training- and testing sets.
    full_trainX, full_trainY, testX, testY = split_train_test(X, y, .75)

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    model = LinearRegression(True)
    loss_stds = []
    loss_means = []

    percents = np.array(range(10, 101))
    for percent in percents:
        losses = []
        for i in range(10):
            ratio = percent / 100
            trainX = full_trainX.sample(round(full_trainX.shape[0] * ratio))
            trainY = full_trainY.loc[trainX.index]
            model.fit(trainX, trainY)
            losses.append(model.loss(testX, testY))

        loss_stds.append(np.std(losses))
        loss_means.append(np.mean(losses))

    loss_means, loss_stds = np.array(loss_means), np.array(loss_stds)
    confidence_top = loss_means + 2 * loss_stds
    confidence_bot = loss_means - 2 * loss_stds

    fig = go.Figure(data=[
        go.Scatter(x=percents, y=loss_means,
                   name='Mean of Loss',
                   mode='lines',
                   marker=dict(color='magenta')),
        go.Scatter(x=percents, y=confidence_bot,
                   fill=None,
                   mode="lines",
                   line=dict(color="lightgrey"),
                   showlegend=False),
        go.Scatter(x=percents, y=confidence_top,
                   fill='tonexty',
                   mode="lines",
                   line=dict(color="lightgrey"),
                   name='Error Ribon')],
        layout=go.Layout(
            title='Decrease of Confidence Interval Size with '
                  'Increase in Size of Train Set',
            xaxis=dict(title='Percents of Train Set Fitted On'),
            yaxis=dict(title='Mean Loss on Test Set')
        ))
    fig.show()
    fig.write_image('.\\ex2_graphs\\Q4.jpeg')
