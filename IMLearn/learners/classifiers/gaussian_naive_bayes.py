from typing import NoReturn
from ...base import BaseEstimator
import numpy as np


class GaussianNaiveBayes(BaseEstimator):
    """
    Gaussian Naive-Bayes classifier
    """

    def __init__(self):
        """
        Instantiate a Gaussian Naive Bayes classifier

        Attributes
        ----------
        self.classes_ : np.ndarray of shape (n_classes,)
            The different labels classes. To be set in `GaussianNaiveBayes.fit`

        self.mu_ : np.ndarray of shape (n_classes,n_features)
            The estimated features means for each class. To be set in `GaussianNaiveBayes.fit`

        self.vars_ : np.ndarray of shape (n_classes, n_features)
            The estimated features variances for each class. To be set in `GaussianNaiveBayes.fit`

        self.pi_: np.ndarray of shape (n_classes)
            The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
        """
        super().__init__()
        self.classes_, self.mu_, self.vars_, self.pi_ = None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a gaussian naive bayes model

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        n_samples, n_features = X.shape
        self.classes_, class_counts = np.unique(y, return_counts=True)
        self.mu_ = np.zeros(len(self.classes_))
        self.vars_ = np.zeros((len(self.classes_), n_features))
        self.pi_ = class_counts / n_samples
        for i, cls in enumerate(self.classes_):
            self.mu_[i] = np.mean(X[y == cls])
            xs_minus_mu = X - np.tile(self.mu_[i], (n_samples, 1))
            self.vars_[i] += np.sum(np.power(xs_minus_mu, 2), axis=0)

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        class_indices = np.argmax(self.likelihood(X), axis=0)
        return self.classes_[class_indices]


    def likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the likelihood of a given data over the estimated model

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data to calculate its likelihood over the different classes.

        Returns
        -------
        likelihoods : np.ndarray of shape (n_samples, n_classes)
            The likelihood for each sample under each of the classes

        """
        if not self.fitted_:
            raise ValueError(
                "Estimator must first be fitted before calling `likelihood` function")

        n_samples, n_features = X.shape
        log_likelihoods = []
        for i, cls in self.classes_:
            first_items = np.log(self.pi_[i]) - n_features * np.log(
                2 * np.pi) / 2 + np.sum(np.log(self.vars_[i]))

            sum_on_x = np.power(X - np.tile(self.mu_[i], (n_samples, 1)), 2)
            sum_on_x = np.divide(sum_on_x, self.vars_[i])
            sum_on_x = np.sum(sum_on_x, axis=1)

            log_like = np.tile(first_items, (n_samples, 1)) - 0.5 * sum_on_x
            log_likelihoods.append()

        return np.array(log_likelihoods).T

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        from ...metrics import misclassification_error
        return misclassification_error(y, self.predict(X))
