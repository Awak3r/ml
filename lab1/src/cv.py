import numpy as np


def kfold_mse(model_factory, X, y, n_splits=5, random_state=42):
    rng = np.random.RandomState(random_state)
    idx = rng.permutation(len(X))
    folds = np.array_split(idx, n_splits)
    mses = []
    for i in range(n_splits):
        val_idx = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(n_splits) if j != i])
        model = model_factory()
        model.fit(X[train_idx], y[train_idx])
        pred = model.predict(X[val_idx])
        mses.append(np.mean((y[val_idx] - pred) ** 2))
    return np.array(mses)


def loo_mse(model_factory, X, y, limit=None):
    n = len(X) if limit is None else min(limit, len(X))
    mses = []
    for i in range(n):
        train_idx = np.concatenate([np.arange(i), np.arange(i + 1, n)])
        model = model_factory()
        model.fit(X[train_idx], y[train_idx])
        pred = model.predict(X[i:i + 1])
        mses.append((y[i] - pred[0]) ** 2)
    return np.array(mses)


def loo_mse_fast(X, y):
    X_b = np.column_stack([np.ones(len(X)), X])
    P = np.linalg.pinv(X_b.T @ X_b)
    fitted = X_b @ P @ X_b.T @ y
    h = np.sum((X_b @ P) * X_b, axis=1)
    return np.mean(((y - fitted) / (1 - h)) ** 2)
