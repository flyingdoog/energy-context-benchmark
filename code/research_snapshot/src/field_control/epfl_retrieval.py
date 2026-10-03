"""Observed-prefix analog baseline; not a reproduction of a physical model."""
import numpy as np


class PrefixRetrieval:
    """Five uniform Euclidean neighbors after training-only feature scaling.

    The library contains complete training episodes. Queries supply prefixes
    only; evaluation targets never enter feature scaling or neighbor selection.
    """
    def fit(self, features, targets, case_ids):
        x = np.asarray(features, dtype=np.float64)
        y = np.asarray(targets, dtype=np.float64)
        ids = np.asarray(case_ids, dtype=str)
        if x.ndim != 2 or y.shape != (len(x), 6) or ids.shape != (len(x),):
            raise ValueError("Expected training features, six targets and case IDs")
        if len(x) < 5 or len(np.unique(ids)) != len(ids):
            raise ValueError("At least five distinct training cases required")
        if not (np.isfinite(x).all() and np.isfinite(y).all()):
            raise ValueError("Complete finite training cases required")
        self.width = x.shape[1]
        self.columns = np.flatnonzero(np.ptp(x, axis=0) > 0)
        self.mean = x.mean(axis=0)
        self.scale = x.std(axis=0)
        self.scale[self.scale < 1e-6] = 1
        # Same binary-control handling as the established Ridge recipe.
        if self.width == 94:
            self.mean[66:94:3] = 0
            self.scale[66:94:3] = 1
        self.library = ((x - self.mean) / self.scale)[:, self.columns]
        self.targets = y.copy()
        self.case_ids = ids.copy()
        return self

    def predict(self, features, batch_size=128):
        x = np.asarray(features, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != self.width or not np.isfinite(x).all():
            raise ValueError("Query prefix shape or finite values differ")
        if batch_size < 1:
            raise ValueError("Positive batch size required")
        z = ((x - self.mean) / self.scale)[:, self.columns]
        predictions, neighbors = [], []
        lib_norm = np.sum(self.library ** 2, axis=1)
        for start in range(0, len(z), batch_size):
            block = z[start:start + batch_size]
            distance = np.maximum(np.sum(block ** 2, axis=1)[:, None] +
                                  lib_norm - 2 * block @ self.library.T, 0)
            index = np.argsort(distance, axis=1, kind="stable")[:, :5]
            predictions.append(self.targets[index].mean(axis=1))
            neighbors.append(self.case_ids[index])
        if not len(z):
            return np.empty((0, 6)), np.empty((0, 5), dtype=str)
        return np.concatenate(predictions), np.concatenate(neighbors)
