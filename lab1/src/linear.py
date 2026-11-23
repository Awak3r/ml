import numpy as np
from scipy.linalg import solve


class MyLinearRegression:
    def __init__(self, method="analytical", lr=0.1, n_iter=3000, batch_size=64,
                 random_state=42):
        self.method = method
        self.lr = lr
        self.n_iter = n_iter
        self.batch_size = batch_size
        self.random_state = random_state
        self.coef_ = None
        self.intercept_ = None
        self.history_ = []

    def _add_bias(self, X):
        return np.column_stack([np.ones(X.shape[0]), X])

    def fit(self, X, y):
        np.random.seed(self.random_state)
        X_b = self._add_bias(X)
        n, d = X_b.shape

        if self.method == "analytical":
            theta = np.linalg.pinv(X_b.T @ X_b) @ X_b.T @ y

        elif self.method == "gd":
            theta = np.zeros(d)
            for i in range(self.n_iter):
                y_pred = X_b @ theta
                grad = (2 / n) * X_b.T @ (y_pred - y)
                theta = theta - self.lr * grad
                self.history_.append(np.mean((y_pred - y) ** 2))

        elif self.method == "sgd":
            theta = np.zeros(d)
            for epoch in range(self.n_iter):
                idx = np.random.permutation(n)
                epoch_loss = 0
                for start in range(0, n, self.batch_size):
                    b = slice(start, min(start + self.batch_size, n))
                    X_batch, y_batch = X_b[idx][b], y[idx][b]
                    y_pred = X_batch @ theta
                    grad = (2 / len(y_batch)) * X_batch.T @ (y_pred - y_batch)
                    theta = theta - self.lr * grad
                    epoch_loss += np.mean((y_pred - y_batch) ** 2) * len(y_batch)
                self.history_.append(epoch_loss / n)
        else:
            raise ValueError("unknown method: " + str(self.method))

        self.intercept_ = theta[0]
        self.coef_ = theta[1:]
        return self

    def predict(self, X):
        X_b = self._add_bias(X)
        theta = np.concatenate([[self.intercept_], self.coef_])
        return X_b @ theta


class MyRegularizedRegression:
    def __init__(self, reg_type="l2", alpha=1.0, p=2, lr=0.05, n_iter=2000):
        self.reg_type = reg_type
        self.alpha = alpha
        self.p = p
        self.lr = lr
        self.n_iter = n_iter
        self.coef_ = None
        self.intercept_ = None

    def _add_bias(self, X):
        return np.column_stack([np.ones(X.shape[0]), X])

    def fit(self, X, y):
        X_b = self._add_bias(X)
        n, d = X_b.shape

        if self.reg_type == "l2":
            I = np.eye(d)
            I[0, 0] = 0
            gram = X_b.T @ X_b + self.alpha * I
            rhs = X_b.T @ y
            if self.alpha > 0:
                theta = solve(gram, rhs, assume_a="pos")
            else:
                theta = np.linalg.pinv(gram) @ rhs
        else:
            theta = np.zeros(d)
            for _ in range(self.n_iter):
                y_pred = X_b @ theta
                grad = (2 / n) * X_b.T @ (y_pred - y)
                theta = theta - self.lr * grad
                w = theta[1:]
                if self.reg_type == "l1":
                    w = np.sign(w) * np.maximum(np.abs(w) - self.lr * self.alpha, 0)
                elif self.reg_type == "elastic":
                    w = w - self.lr * self.alpha * w
                    w = np.sign(w) * np.maximum(np.abs(w) - self.lr * self.alpha, 0)
                elif self.reg_type == "lp":
                    w = w - self.lr * self.alpha * self.p * np.sign(w) * np.abs(w) ** (self.p - 1)
                else:
                    raise ValueError("unknown reg_type: " + str(self.reg_type))
                theta[1:] = w

        self.intercept_ = theta[0]
        self.coef_ = theta[1:]
        return self

    def predict(self, X):
        X_b = self._add_bias(X)
        theta = np.concatenate([[self.intercept_], self.coef_])
        return X_b @ theta
