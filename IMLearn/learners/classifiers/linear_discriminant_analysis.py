from typing import NoReturn
from ...base import BaseEstimator
import numpy as np
from numpy.linalg import det, inv
from ...metrics import misclassification_error


class LDA(BaseEstimator):
    """
    Linear Discriminant Analysis (LDA) classifier

    Attributes
    ----------
    self.classes_ : np.ndarray of shape (n_classes,)
        The different labels classes. To be set in `LDA.fit`

    self.mu_ : np.ndarray of shape (n_classes,n_features)
        The estimated features means for each class. To be set in `LDA.fit`

    self.cov_ : np.ndarray of shape (n_features,n_features)
        The estimated features covariance. To be set in `LDA.fit`

    self._cov_inv : np.ndarray of shape (n_features,n_features)
        The inverse of the estimated features covariance. To be set in `LDA.fit`

    self.pi_: np.ndarray of shape (n_classes)
        The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
    """

    def __init__(self):
        """
        Instantiate an LDA classifier
        """
        super().__init__()
        self.classes_, self.mu_, self.cov_, self._cov_inv, self.pi_ = None, None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits an LDA model.
        Estimates gaussian for each label class - Different mean vector, same covariance
        matrix with dependent features.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        n_samples, n_features = X.shape
        self.classes_, class_counts = np.unique(y, return_counts=True)
        k = self.classes_.shape[0]

        self.pi_ = class_counts / n_samples

        self.mu_ = np.zeros((class_counts, n_features))
        for i, cls in enumerate(self.classes_):
            self.mu_[i] += np.sum(X[y == cls]) / class_counts[i]

        self.cov_ = np.zeros(k, k)
        for sample, response in X, y:
            x_minus_mu = sample - self.mu_[self.classes_ == y][0]
            self.cov_ += np.outer(x_minus_mu, x_minus_mu)

        self.cov_ /= n_samples
        self._cov_inv = inv(self.cov_)  # todo maybe switch it to pinv

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
        aks = np.matmul(self.mu_, self._cov_inv.T)
        cov_inv_times_mu = np.matmul(self._cov_inv, self.mu_.T).T
        # ^The i'th row is cov_inv multiplied by x_i
        bks = (cov_inv_times_mu * self._cov_inv).sum(-1)
        # ^actually, we the i'th value in the vector this the dot product of the
        # i'th rows in the matrices cov_inv_times_mu & self._cov_inv

        akt_times_x = np.matmul(X, aks.T)
        args_rows_to_max = akt_times_x + np.tile(bks, (X.shape[0], 1))
        class_inds = np.argmax(args_rows_to_max,
                               axis=1)  # find argmax for each row

        return self.classes_[class_inds]

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
        for yi, cls in enumerate(self.classes_):
            first_part = np.log(self.pi_[yi]) - \
                       n_features * np.log(2 * np.pi) / 2 - np.log(
                det(self.cov_)) / 2

            x_minus_mu = (X - np.tile(self.mu_[yi], (n_samples, 1)))
            left_mul = np.matmul(self._cov_inv, x_minus_mu.T)
            mat_mul = (x_minus_mu.T * left_mul).sum(0)
            log_like_yi = first_part - 0.5*mat_mul
            log_likelihoods.append(log_like_yi)

        log_likelihoods = np.array(log_likelihoods)
        return log_likelihoods.T

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
        return misclassification_error(y, self.predict(X))
