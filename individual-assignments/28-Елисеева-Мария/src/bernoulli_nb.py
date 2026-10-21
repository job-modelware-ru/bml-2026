"""Собственная реализация Bernoulli Naive Bayes и бинаризатора признаков.

Используется только NumPy; от scikit-learn берутся лишь базовые классы,
чтобы модели можно было подставлять в cross_val_score и Pipeline.
"""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin


class MedianBinarizer(TransformerMixin, BaseEstimator):
    """Бинаризация: порог каждого признака = медиана на обучающей выборке.

    x_ij > median_j -> 1, иначе 0. Медианы считаются только в fit.
    """

    def fit(self, X, y=None):
        self.thresholds_ = np.median(np.asarray(X, dtype=float), axis=0)
        return self

    def transform(self, X):
        return (np.asarray(X, dtype=float) > self.thresholds_).astype(np.float64)


class BernoulliNaiveBayes(ClassifierMixin, BaseEstimator):
    """Наивный байесовский классификатор для бинарных признаков.

    P(x | y=k) = prod_j theta_kj^x_j * (1 - theta_kj)^(1 - x_j)
    theta_kj = (N_kj + alpha) / (N_k + 2 * alpha)      (сглаживание Лапласа)

    alpha     - параметр сглаживания (должен быть > 0);
    threshold - если задан, вход бинаризуется как x > threshold;
    fit_prior - оценивать ли априорные вероятности классов по данным.
    """

    def __init__(self, alpha=1.0, threshold=None, fit_prior=True):
        self.alpha = alpha
        self.threshold = threshold
        self.fit_prior = fit_prior

    def _binarize(self, X):
        X = np.asarray(X, dtype=float)
        if self.threshold is not None:
            X = (X > self.threshold).astype(np.float64)
        return X

    def fit(self, X, y):
        X = self._binarize(X)
        y = np.asarray(y)
        if not np.isin(X, (0.0, 1.0)).all():
            raise ValueError("Признаки должны быть бинарными (0/1); задайте threshold или бинаризуйте данные.")
        if self.alpha <= 0:
            raise ValueError("alpha должен быть > 0")

        self.classes_ = np.unique(y)
        n_classes, n_features = len(self.classes_), X.shape[1]
        self.n_features_in_ = n_features

        class_count = np.zeros(n_classes)
        feature_count = np.zeros((n_classes, n_features))
        for idx, c in enumerate(self.classes_):
            Xc = X[y == c]
            class_count[idx] = Xc.shape[0]
            feature_count[idx] = Xc.sum(axis=0)

        self.class_count_ = class_count
        self.feature_count_ = feature_count
        self.theta_ = (feature_count + self.alpha) / (class_count[:, None] + 2.0 * self.alpha)  # P(x_j=1 | y=k)
        if self.fit_prior:
            self.class_log_prior_ = np.log(class_count / class_count.sum())
        else:
            self.class_log_prior_ = np.full(n_classes, -np.log(n_classes))
        return self

    def _joint_log_likelihood(self, X):
        X = self._binarize(X)
        log_theta = np.log(self.theta_)
        log_neg_theta = np.log1p(-self.theta_)
        # sum_j [x_j log th + (1-x_j) log(1-th)] = X @ (log th - log(1-th)) + sum_j log(1-th)
        return X @ (log_theta - log_neg_theta).T + log_neg_theta.sum(axis=1) + self.class_log_prior_

    def predict_log_proba(self, X):
        jll = self._joint_log_likelihood(X)
        m = jll.max(axis=1, keepdims=True)
        return jll - (m + np.log(np.exp(jll - m).sum(axis=1, keepdims=True)))

    def predict_proba(self, X):
        return np.exp(self.predict_log_proba(X))

    def predict(self, X):
        return self.classes_[np.argmax(self._joint_log_likelihood(X), axis=1)]
