"""Fixed marginal and cross-block moments for a whole-group reference."""
import numpy as np


class GroupMomentMap:
    def __init__(self, normalizer, protected, moved, width=16, seed=2027):
        self.normalizer = normalizer
        d = len(normalizer.mean)
        self.protected = np.r_[protected, np.asarray(protected) + d].astype(int)
        self.moved = np.r_[moved, np.asarray(moved) + d].astype(int)
        rng = np.random.default_rng(seed)
        self.left = rng.choice(np.array([-1., 1.], dtype=np.float32),
            size=(len(self.protected), width)) / np.sqrt(len(self.protected))
        self.right = rng.choice(np.array([-1., 1.], dtype=np.float32),
            size=(len(self.moved), width)) / np.sqrt(len(self.moved))
        self.width = width

    def group(self, values, permutations):
        x = self.normalizer.transform(values)
        base = np.r_[np.log1p(len(x)), x.mean(axis=0), x.std(axis=0)]
        u = x[:, self.protected] @ self.left
        v = x[:, self.moved] @ self.right
        u = u - u.mean(axis=0)
        v = v - v.mean(axis=0)
        result = {'Marginal': np.r_[base, np.zeros(self.width**2)]}
        result['Paired'] = np.r_[base, (u.T @ v / len(x)).ravel()]
        for name, order in permutations.items():
            order = np.asarray(order, dtype=int)
            if not np.array_equal(np.sort(order), np.arange(len(x))):
                raise ValueError('Expected a within-group record permutation')
            result[name] = np.r_[base, (u.T @ v[order] / len(x)).ravel()]
        return {name: value.astype(np.float32) for name, value in result.items()}


def encode_groups(inputs, mapper, donors):
    values = {}
    for rows in inputs.rows:
        # Reuse the original cached mappings, whose donors remain in this group.
        reverse = {int(row): i for i, row in enumerate(rows)}
        permutations = {name: [reverse[int(donor[row])] for row in rows]
                        for name, donor in donors.items()}
        encoded = mapper.group(inputs.features[rows], permutations)
        for name, vector in encoded.items():
            values.setdefault(name, []).append(vector)
    return {name: np.stack(vectors) for name, vectors in values.items()}
