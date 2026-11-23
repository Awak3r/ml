"""Simplified two-branch model based on liptonMartin's MAI notebook:
https://github.com/liptonMartin/mai_labs_machine_learning (lab_1/01_train_sklearn.ipynb).
The branch architecture and input columns are retained; numeric cut points are
estimated in fit. Each branch is OLS, but the weighted blend is not a single OLS.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

BUCKET_COLUMNS = ['CreditScore', 'DebtToIncomeRatio', 'CreditCardUtilizationRate',
                  'LengthOfCreditHistory', 'AnnualIncome', 'NetWorth']
EDUCATION_ORDER = {'High School': 0, 'Associate': 1, 'Bachelor': 2, 'Master': 3, 'Doctorate': 4}
WEIGHT_COLUMNS = ['TotalDebtToIncomeRatio', 'MonthlyIncome', 'InterestRate', 'NetWorth',
                  'CreditScore', 'LoanAmount', 'LengthOfCreditHistory', 'BankruptcyHistory',
                  'Age', 'PreviousLoanDefaults']


def main_features(frame, cuts):
    """Six bucket numbers and three flags; each step in a bucket has the same weight."""
    features = {name: sum(frame[name].ge(t) for t in thresholds)
                for name, thresholds in cuts.items()}
    features['Bankruptcy'] = frame.BankruptcyHistory.fillna(0.0)
    features['PreviousDefaults'] = frame.PreviousLoanDefaults.fillna(0.0)
    features['NotEmployed'] = frame.EmploymentStatus.ne('Employed')
    return pd.DataFrame(features, index=frame.index).to_numpy(dtype=float)


def _column_cuts(values, response):
    """Find steps in a partial response sorted by one predictor."""
    n = len(values)
    prefix = np.r_[0, np.cumsum(response)]
    candidates = np.flatnonzero(values[:-1] != values[1:]) + 1

    def best_split(lo, hi):
        p = candidates[(candidates >= lo + 20) & (candidates <= hi - 20)]
        if not len(p):
            return 0.0, None
        # Reduction in squared error from separate left/right means.
        diff = (prefix[p] - prefix[lo]) / (p - lo) - (prefix[hi] - prefix[p]) / (hi - p)
        gain = (p - lo) * (hi - p) / (hi - lo) * diff ** 2
        k = gain.argmax()
        return gain[k], p[k]

    bounds = [0, n]
    min_gain = 3 * np.log(n) * np.var(response)
    for _ in range(5):
        gain, p = max((best_split(lo, hi) for lo, hi in zip(bounds, bounds[1:])), key=lambda s: s[0])
        if p is None or gain <= min_gain:
            break
        bounds = sorted([*bounds, p])
    # Revisit early choices now that their neighboring boundaries are known.
    for _ in range(3):
        for i in range(1, len(bounds) - 1):
            _, bounds[i] = best_split(bounds[i - 1], bounds[i + 1])
    return values[bounds[1:-1]]


def index_features(frame):
    return frame[WEIGHT_COLUMNS].assign(AgeSquared=frame.Age ** 2,
                                        EducationOrder=frame.EducationLevel.map(EDUCATION_ORDER))


class CompleteRowsOnly:
    """Exclude incomplete training examples, but predict on every evaluation row."""

    def __init__(self, inner):
        self.inner = inner

    def fit(self, frame):
        keep = frame.drop(columns=['RiskScore', 'ID'], errors='ignore').notna().all(axis=1)
        self.inner.fit(frame.loc[keep])
        self.n_features_ = self.inner.n_features_
        return self

    def predict(self, frame):
        return self.inner.predict(frame)


class BranchMixtureRegression:
    def __init__(self, use_branches=True):
        self.use_branches = use_branches

    @staticmethod
    def _assign_branches(x, y):
        """Alternate OLS fits and assignment to the branch with the smaller error."""
        high = y > np.median(y)
        if min(high.sum(), (~high).sum()) < 2:
            raise ValueError('Cannot split the sample into two branches')
        for _ in range(50):
            low_pred = LinearRegression().fit(x[~high], y[~high]).predict(x)
            high_pred = LinearRegression().fit(x[high], y[high]).predict(x)
            updated = np.abs(y - high_pred) < np.abs(y - low_pred)
            if np.array_equal(updated, high) or min(updated.sum(), (~updated).sum()) < 2:
                break
            high = updated
        return high if high_pred.mean() >= low_pred.mean() else ~high

    def _learn_cuts(self, frame, y):
        cuts = {c: frame[c].quantile([.2, .4, .6, .8]).unique() for c in BUCKET_COLUMNS}
        order = {c: np.argsort(frame[c].to_numpy()) for c in BUCKET_COLUMNS}
        for _ in range(5):
            x = main_features(frame, cuts)
            high = self._assign_branches(x, y)
            low_fit, high_fit = (LinearRegression().fit(x[m], y[m]) for m in (~high, high))
            residual = y - np.where(high, high_fit.predict(x), low_fit.predict(x))
            # Limit the influence of outliers on cut selection, not on final OLS.
            mad = np.median(np.abs(residual - np.median(residual)))
            residual = residual.clip(-4 * mad, 4 * mad)
            for j, c in enumerate(BUCKET_COLUMNS):
                weight = np.where(high, high_fit.coef_[j], low_fit.coef_[j])
                partial = residual + weight * x[:, j]
                idx = order[c]
                cuts[c] = _column_cuts(frame[c].to_numpy()[idx], partial[idx])
        return cuts

    def fit(self, frame):
        y = frame.RiskScore.to_numpy(dtype=float)
        self.cuts_ = self._learn_cuts(frame, y)
        x = main_features(frame, self.cuts_)
        self.n_features_ = x.shape[1]
        if not self.use_branches:
            self.single_ = LinearRegression().fit(x, y)
            return self

        high = self._assign_branches(x, y)
        self.low_ = LinearRegression().fit(x[~high], y[~high])
        self.high_ = LinearRegression().fit(x[high], y[high])

        design = index_features(frame)
        self.index_median_ = design.median()
        self.scaler_ = StandardScaler()
        scaled = self.scaler_.fit_transform(design.fillna(self.index_median_))
        window = np.ones(len(y), dtype=bool)
        for _ in range(6):
            self.index_ = LinearRegression().fit(scaled[window], high[window].astype(float))
            p = self.index_.predict(scaled)
            window = (p > 0) & (p < 1)
            if window.sum() < scaled.shape[1] + 1:
                break
        return self

    def predict(self, frame):
        if frame.empty:
            return np.empty(0)
        x = main_features(frame, self.cuts_)
        if not self.use_branches:
            return self.single_.predict(x)
        design = index_features(frame)
        scaled = self.scaler_.transform(design.fillna(self.index_median_))
        p = self.index_.predict(scaled).clip(0, 1)
        return (1 - p) * self.low_.predict(x) + p * self.high_.predict(x)
