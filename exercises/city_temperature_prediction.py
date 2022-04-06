import plotly.colors

import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio

pio.templates.default = "simple_white"

CSV_FILENAME = "..\\datasets\\City_Temperature.csv"


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    parser = lambda s: pd.to_datetime(s, errors='coerce')
    df = pd.read_csv(filename, parse_dates=['Date'], date_parser=parser)
    df['DayOfYear'] = df['Date'].dt.dayofyear
    df = df[df['Temp'] >= -50]
    return df.drop(['Temp'], axis=1), df['Temp']


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    X, y = load_data(CSV_FILENAME)

    # Question 2 - Exploring data for specific country
    israel_X = X[X['Country'] == 'Israel']
    israel_y = y[israel_X.index]
    israel = pd.concat([israel_X['DayOfYear'],
                        israel_X['Year'].astype(str),
                        israel_X['Month'],
                        israel_y.rename('Temp')], axis=1)

    fig = px.scatter(israel, x="DayOfYear", y="Temp", color="Year",
                     title="Temperatures by Day of Year in Israel")
    fig.write_image(".\\ex2_graphs\\israel_temps.jpeg")

    std_by_month = israel.groupby(['Month'])['Temp'].std().rename(
        "TemperatureSTD")
    fig = px.bar(std_by_month, x=std_by_month.index, y='TemperatureSTD')
    fig.write_image(".\\ex2_graphs\\israel_by_month.jpeg")

    # Question 3 - Exploring differences between countries
    df = pd.concat([X, y], axis=1)
    by_country_by_month = df.groupby(['Country', 'Month'])
    byby_mean = by_country_by_month['Temp'].mean()
    byby_std = by_country_by_month['Temp'].std()
    fig = px.line(x=byby_mean.index.get_level_values(1), y=byby_mean,
                  error_y=byby_std, color=byby_mean.index.get_level_values(0))
    fig.update_layout(xaxis=dict(title='Month'),
                      yaxis=dict(title=' Average Temperature'),
                      title='Monthly Average Temperature by Country')
    fig.write_image(".\\ex2_graphs\\by_country_by_month.jpeg")

    # Question 4 - Fitting model for different values of `k`
    X_train, y_train, X_test, y_test = split_train_test(israel_X['DayOfYear'],
                                                        israel_y, .75)

    degrees = np.array(range(1, 11))
    losses = []
    for k in degrees:
        model = PolynomialFitting(k)
        model.fit(X_train, y_train)
        losses.append(np.round(model.loss(X_test, y_test), 2))

    fig = px.line(x=degrees, y=losses, markers=True)
    fig.update_layout(
        xaxis={'title': 'Polynomial Degree'},
        yaxis={"title": 'Loss'},
        title='Polynomial Fit - Change in Loss for Change in Model Degree')
    fig.update_traces(marker={'color': '#b30a69'})
    fig.write_image('.\\ex2_graphs\\loss_vs_deg.jpeg')

    # Question 5 - Evaluating fitted model on different countries
    OPTIMAL_K = 5
    model = PolynomialFitting(5)
    model.fit(israel_X['DayOfYear'], israel_y)
    country_loss = df.groupby('Country').apply(
        lambda x: model.loss(x['DayOfYear'], x['Temp']))
    fig = px.bar(country_loss,
                 title=f'Polynomial of Degree={OPTIMAL_K} '
                       f'Fitted on Israel - Loss by Country')
    fig.update_layout(xaxis=dict(title='Country'),
                      yaxis=dict(title='Loss'),
                      showlegend=False)
    fig.write_image('.\\ex2_graphs\\loss_per_country.jpeg')