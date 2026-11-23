import numpy as np


class ZScoreScaler:
    def fit(self, X):
        X = np.asarray(X, dtype=float)
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        self.std_[self.std_ == 0] = 1
        return self

    def transform(self, X):
        return (np.asarray(X, dtype=float) - self.mean_) / self.std_

    def fit_transform(self, X):
        return self.fit(X).transform(X)


class MinMaxScaler:
    def fit(self, X):
        X = np.asarray(X, dtype=float)
        self.min_ = X.min(axis=0)
        self.max_ = X.max(axis=0)
        rng = self.max_ - self.min_
        rng[rng == 0] = 1
        return self

    def transform(self, X):
        return (np.asarray(X, dtype=float) - self.min_) / (self.max_ - self.min_)

    def fit_transform(self, X):
        return self.fit(X).transform(X)
