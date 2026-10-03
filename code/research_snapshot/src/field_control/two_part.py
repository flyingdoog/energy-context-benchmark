"""Fixed LightGBM occurrence and positive-magnitude forecasting reference."""
import json
from pathlib import Path
import time
import numpy as np
from lightgbm import Booster, LGBMClassifier, LGBMRegressor

PARAMETERS = dict(n_estimators=100, num_leaves=31, n_jobs=2,
                  random_state=2027, verbosity=-1)


class TwoPartLead:
    """One future bin. Fit accepts training data only, never validation labels."""
    @classmethod
    def fit(cls, x_train, y_train, folder):
        x = np.asarray(x_train)
        y = np.asarray(y_train, dtype=float)
        if x.ndim != 2 or y.shape != (len(x),) or len(x) == 0:
            raise ValueError("Nonempty aligned training data required")
        if np.isinf(x).any() or not np.isfinite(y).all() or (y < 0).any():
            raise ValueError("Invalid training values; NaN feature masks remain supported")
        folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
        receipt = folder / "receipt.json"
        if receipt.exists() and json.loads(receipt.read_text()).get("completed"):
            model = cls.load(folder)
            if model.info["train_cases"] != len(x) or model.info["features"] != x.shape[1]:
                raise ValueError("Cached fit dimensions differ")
            model.new_fits = 0
            return model
        started = time.monotonic(); positive = y > 0
        prevalence = float(positive.mean()); new_fits = 0
        classifier = folder / "occurrence.txt"
        regressor = folder / "magnitude.txt"
        constant_p = prevalence if prevalence in (0., 1.) else None
        if constant_p is None and not classifier.exists():
            model = LGBMClassifier(**PARAMETERS)
            model.fit(x, positive.astype(int)); model.booster_.save_model(str(classifier))
            new_fits += 1
        if positive.any() and not regressor.exists():
            model = LGBMRegressor(**PARAMETERS)
            model.fit(x[positive], y[positive]); model.booster_.save_model(str(regressor))
            new_fits += 1
        info = dict(completed=True, train_cases=len(x), positive_cases=int(positive.sum()),
                    features=x.shape[1], prevalence=prevalence, constant_probability=constant_p,
                    no_positive_targets=not bool(positive.any()), parameters=PARAMETERS,
                    component_models=int(constant_p is None)+int(positive.any()),
                    new_fits=new_fits, seconds=time.monotonic()-started,
                    recipe="Binary occurrence plus positive-only MSE regression; fixed defaults, no HPO")
        receipt.write_text(json.dumps(info, indent=2)+"\n", encoding="utf-8")
        model = cls.load(folder); model.new_fits = new_fits
        return model

    @classmethod
    def load(cls, folder):
        model = cls(); folder = Path(folder)
        model.info = json.loads((folder / "receipt.json").read_text())
        if not model.info.get("completed"): raise ValueError("Incomplete two-part fit")
        model.classifier = None if model.info["constant_probability"] is not None else Booster(model_file=str(folder/"occurrence.txt"))
        model.regressor = None if model.info["no_positive_targets"] else Booster(model_file=str(folder/"magnitude.txt"))
        model.new_fits = 0
        return model

    def predict(self, x):
        x = np.asarray(x)
        if x.ndim != 2 or x.shape[1] != self.info["features"] or np.isinf(x).any():
            raise ValueError("Query feature layout differs")
        if self.classifier is None:
            p = np.full(len(x), self.info["constant_probability"], dtype=float)
        else: p = self.classifier.predict(x, num_threads=2)
        m = np.zeros(len(x)) if self.regressor is None else np.maximum(self.regressor.predict(x, num_threads=2), 0)
        if not (np.isfinite(p).all() and np.isfinite(m).all() and ((p>=0)&(p<=1)).all()):
            raise ValueError("Invalid occurrence probability or conditional mean")
        return p, m, p*m
