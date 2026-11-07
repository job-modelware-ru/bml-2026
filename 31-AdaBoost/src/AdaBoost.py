"""AdaBoost для бинарной классификации — реализация с нуля (только numpy).

Метки приводим к {-1, +1}:
  eps_t   = sum_i w_i * [h_t(x_i) != y_i]          взвешенная ошибка
  alpha_t = 0.5 * ln((1 - eps_t) / eps_t)           вес слабого классификатора
  w_i    <- w_i * exp(-alpha_t * y_i * h_t(x_i)) / Z_t,  Z_t = 2*sqrt(eps_t*(1-eps_t))
  H(x)    = sign(sum_t alpha_t * h_t(x))
Это пошаговая минимизация экспоненциальной потери exp(-y * F(x)).
"""
import numpy as np


class WeightedStump:
    """Решающий пень: предсказывает pol, если x[j] > thr, иначе -pol.
    Выбирает (j, thr, pol) с минимальной ВЗВЕШЕННОЙ ошибкой."""

    def fit(self, X, y, w):
        best = np.inf
        for j in range(X.shape[1]):
            order = np.argsort(X[:, j])
            xs, ys, ws = X[order, j], y[order], w[order]
            pos_left = np.cumsum(ws * (ys == 1))                      # +1 слева от порога — ошибки
            neg_right = np.sum(ws * (ys == -1)) - np.cumsum(ws * (ys == -1))
            err_pos = pos_left + neg_right                            # полярность +1
            valid = np.append(xs[1:] != xs[:-1], False)               # порог только между разными x
            for pol, err in ((1, err_pos), (-1, 1 - err_pos)):
                e = np.where(valid, err, np.inf)
                k = int(np.argmin(e))
                if e[k] < best:
                    best = e[k]
                    self.j, self.pol = j, pol
                    self.thr = (xs[k] + xs[k + 1]) / 2
        return self

    def predict(self, X):
        return np.where(X[:, self.j] > self.thr, self.pol, -self.pol)


class MyAdaBoost:
    def __init__(self, n_estimators=100):
        self.T = n_estimators

    def fit(self, X, y01):
        y = np.where(y01 == 1, 1, -1)
        n = len(y)
        w = np.full(n, 1 / n)
        self.models, self.alphas = [], []
        self.history = {"eps": [], "Z": [], "train_err": [], "exp_loss": []}
        F = np.zeros(n)
        for _ in range(self.T):
            stump = WeightedStump().fit(X, y, w)
            pred = stump.predict(X)
            eps = np.clip(w[pred != y].sum(), 1e-10, 1 - 1e-10)
            if eps >= 0.5:                      # хуже случайного — останавливаемся
                break
            alpha = 0.5 * np.log((1 - eps) / eps)
            w = w * np.exp(-alpha * y * pred)
            w /= w.sum()
            self.models.append(stump)
            self.alphas.append(alpha)
            F += alpha * pred
            self.history["eps"].append(eps)
            self.history["Z"].append(2 * np.sqrt(eps * (1 - eps)))
            self.history["train_err"].append(np.mean(np.sign(F) != y))
            self.history["exp_loss"].append(np.mean(np.exp(-y * F)))
        return self

    def decision_function(self, X, k=None):
        k = len(self.models) if k is None else k
        return sum(a * m.predict(X) for a, m in zip(self.alphas[:k], self.models[:k]))

    def predict(self, X, k=None):
        return (self.decision_function(X, k) > 0).astype(int)
