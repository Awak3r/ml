import numpy as np


def mse_metric(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def mae_metric(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


def r2_metric(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot


def mape_metric(y_true, y_pred):
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
