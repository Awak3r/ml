"""Leakage-free cross-validation shared by the notebook and the model modules.

The notebook imports `training_rows` and `cross_validate` from here. Nothing in
this module writes files or parses command-line arguments.
"""
import json

import numpy as np

TARGET_MSE = 19.0


def training_rows(frame):
    """Discard duplicated rows and absent/technical targets, never impute y."""
    result = frame.drop_duplicates()
    return result.loc[result.RiskScore.between(0, 100)].copy()


def evaluate_rows(frame, prediction):
    """MSE, MAE and R2 overall and split by completeness of the predictor row."""
    y = frame.RiskScore.to_numpy(dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    if prediction.shape != y.shape or not np.isfinite(prediction).all():
        raise ValueError('Predictions must be a finite vector matching the target')
    error = (y - prediction) ** 2
    complete = frame.drop(columns=['RiskScore', 'ID'], errors='ignore').notna().all(axis=1).to_numpy()
    result = {'rows': len(frame), 'mse': float(error.mean()),
              'mae': float(np.mean(np.abs(y - prediction))),
              'r2': float(1 - error.sum() / np.sum((y - y.mean()) ** 2))}
    for name, mask in [('complete', complete), ('incomplete', ~complete)]:
        result[name + '_rows'] = int(mask.sum())
        result[name + '_mse'] = float(error[mask].mean()) if mask.any() else None
    return result


def cross_validate(frame, model_factory, n_splits=5, seed=314159):
    """Out-of-fold predictions; a model never sees its validation rows."""
    if not 2 <= n_splits <= len(frame):
        raise ValueError('folds must be between 2 and the number of training rows')
    folds = np.array_split(np.random.default_rng(seed).permutation(len(frame)), n_splits)
    oof = np.empty(len(frame), dtype=float)
    scores = []
    for number, valid_indices in enumerate(folds, 1):
        train_indices = np.concatenate([part for i, part in enumerate(folds, 1) if i != number])
        training, validation = frame.iloc[train_indices], frame.iloc[valid_indices]
        model = model_factory().fit(training)
        prediction = model.predict(validation.drop(columns='RiskScore'))
        oof[valid_indices] = prediction
        score = evaluate_rows(validation, prediction)
        score.update(fold=number, features=model.n_features_)
        scores.append(score)
        print(json.dumps(score, ensure_ascii=False), flush=True)
    summary = evaluate_rows(frame, oof)
    summary['fold_mse_mean'] = float(np.mean([s['mse'] for s in scores]))
    summary['fold_mse_std'] = float(np.std([s['mse'] for s in scores]))
    summary['below_19_on_oof'] = summary['mse'] < TARGET_MSE
    return {'folds': scores, 'oof': summary}
