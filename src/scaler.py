"""scaler.py -- minimal feature standardization, saved alongside the model
so predict.py applies the exact same transform used at training time."""

import numpy as np


class StandardScaler:
    def __init__(self):
        self.mean_ = None
        self.scale_ = None

    def fit(self, X):
        self.mean_ = X.mean(axis=0)
        std = X.std(axis=0)
        std[std == 0] = 1.0  # avoid divide-by-zero on constant columns
        self.scale_ = std
        return self

    def transform(self, X):
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def save(self, path):
        np.savez(path, mean=self.mean_, scale=self.scale_)

    @classmethod
    def load(cls, path):
        data = np.load(path)
        scaler = cls()
        scaler.mean_ = data["mean"]
        scaler.scale_ = data["scale"]
        return scaler
