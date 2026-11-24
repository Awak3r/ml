from math import ceil
from numbers import Integral, Real

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted


def _split_candidates(feature, target):
    order = np.argsort(feature)
    values, target = feature[order], target[order]
    cuts = np.flatnonzero(values[:-1] < values[1:]) + 1
    if not cuts.size:
        return np.array([]), np.array([]), cuts

    n = values.size
    left, right = cuts, n - cuts
    counts = np.cumsum(target == target[0], dtype=float)
    pos_left = counts[cuts - 1]
    pos_right = counts[-1] - pos_left
    ginis = -(2.0 / n) * (
        pos_left * (left - pos_left) / left + pos_right * (right - pos_right) / right
    )

    lower, upper = values[cuts - 1], values[cuts]
    thresholds = lower / 2 + upper / 2
    thresholds = np.where(thresholds > lower, thresholds, upper)
    return thresholds, ginis, cuts


def _best_index(ginis):
    return np.flatnonzero(ginis >= ginis.max() - 2 * np.finfo(float).eps)[0]


def find_best_split(feature_vector, target_vector):
    feature = np.asarray(feature_vector, dtype=float)
    target = np.asarray(target_vector)
    if feature.ndim != 1 or target.ndim != 1 or feature.shape != target.shape:
        raise ValueError("feature_vector and target_vector must be equally sized vectors")
    thresholds, ginis, _ = _split_candidates(feature, target)
    if not thresholds.size:
        return thresholds, ginis, None, None
    best = _best_index(ginis)
    return thresholds, ginis, thresholds[best], ginis[best]


def _sample_limit(value, n_samples, minimum, name):
    if value is None:
        return minimum
    if isinstance(value, Integral) and not isinstance(value, bool):
        if value >= minimum:
            return int(value)
    elif isinstance(value, Real) and 0 < value <= 1:
        return max(minimum, ceil(value * n_samples))
    raise ValueError(f"{name} must be an integer >= {minimum} or a fraction in (0, 1]")


class DecisionTree(ClassifierMixin, BaseEstimator):
    def __init__(self, feature_types, max_depth=None, min_samples_split=None, min_samples_leaf=None):
        self.feature_types = feature_types
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf

    def _fit_node(self, sub_X, sub_y, node, depth=0):
        n = len(sub_y)
        ones = np.count_nonzero(sub_y)
        node.update(type="terminal", **{"class": self.classes_[int(2 * ones > n)]})
        if (
            ones in (0, n)
            or (self.max_depth is not None and depth >= self.max_depth)
            or n < max(self._min_samples_split, 2 * self._min_samples_leaf)
        ):
            return

        best_gini = -np.inf
        best = None
        for feature, feature_type in enumerate(self.feature_types):
            column = sub_X[:, feature]
            if feature_type == "real":
                values = column.astype(float, copy=False)
            else:
                categories, codes = np.unique(column, return_inverse=True)
                counts = np.bincount(codes)
                rates = np.bincount(codes, weights=sub_y) / counts
                order = np.argsort(-rates, kind="stable")
                ranks = np.empty(len(order), dtype=int)
                ranks[order] = np.arange(len(order))
                values = ranks[codes]

            thresholds, ginis, sizes = _split_candidates(values, sub_y)
            valid = np.flatnonzero((sizes >= self._min_samples_leaf) & (n - sizes >= self._min_samples_leaf))
            if not valid.size:
                continue
            idx = valid[_best_index(ginis[valid])]
            if ginis[idx] > best_gini:
                threshold = thresholds[idx]
                if feature_type == "real":
                    rule = {"threshold": threshold}
                else:
                    rule = {"categories_split": frozenset(categories[ranks < threshold])}
                best_gini = ginis[idx]
                best = feature, rule, values < threshold

        if best is None:
            return
        feature, rule, left = best
        node.clear()
        node.update(type="nonterminal", feature_split=feature, **rule, left_child={}, right_child={})
        self._fit_node(sub_X[left], sub_y[left], node["left_child"], depth + 1)
        self._fit_node(sub_X[~left], sub_y[~left], node["right_child"], depth + 1)

    def _predict_node(self, x, node):
        while node["type"] == "nonterminal":
            value = x[node["feature_split"]]
            if "threshold" in node:
                left = float(value) < node["threshold"]
            else:
                left = value in node["categories_split"]
            node = node["left_child"] if left else node["right_child"]
        return node["class"]

    def fit(self, X, y):
        X, y = np.asarray(X), np.asarray(y)
        if X.ndim != 2 or y.ndim != 1 or not len(y) or len(X) != len(y):
            raise ValueError("Expected nonempty 2D X and 1D y with the same number of rows")
        if len(self.feature_types) != X.shape[1] or any(
            kind not in ("real", "categorical") for kind in self.feature_types
        ):
            raise ValueError("Expected one 'real' or 'categorical' feature type for each column")
        if self.max_depth is not None and (
            not isinstance(self.max_depth, Integral) or isinstance(self.max_depth, bool) or self.max_depth < 1
        ):
            raise ValueError("max_depth must be a positive integer or None")

        classes, encoded = np.unique(y, return_inverse=True)
        if len(classes) > 2:
            raise ValueError("DecisionTree supports binary classification only")
        self._min_samples_split = _sample_limit(self.min_samples_split, len(y), 2, "min_samples_split")
        self._min_samples_leaf = _sample_limit(self.min_samples_leaf, len(y), 1, "min_samples_leaf")
        self.classes_ = classes
        self.n_features_in_ = X.shape[1]
        self._tree = {}
        self._fit_node(X, encoded, self._tree)
        return self

    def predict(self, X):
        check_is_fitted(self, "_tree")
        X = np.asarray(X)
        if X.ndim != 2 or X.shape[1] != self.n_features_in_:
            raise ValueError(f"Expected 2D X with {self.n_features_in_} columns")
        return np.array([self._predict_node(x, self._tree) for x in X], dtype=self.classes_.dtype)
