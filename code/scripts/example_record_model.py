"""Small synthetic example of supplying a predictor to the four-arm API."""
import argparse
import json
from pathlib import Path
import run_local
import numpy as np
import pandas as pd
from field_control.record_linkage import donor_mapping
from field_control.record_comparison import RecordInputView, fit_record_models, predict_record_models
from field_control.record_reference_inputs import SEEDS


class LinearExample:
    """Usage example only: finite-value linear least squares, no search."""
    def __init__(self, seed=2027):
        self.seed = seed

    @staticmethod
    def design(x):
        return np.c_[np.ones(len(x)), np.nan_to_num(x, nan=0., posinf=0., neginf=0.)]

    def fit(self, x, y):
        self.coef = np.linalg.lstsq(self.design(x), y, rcond=None)[0]
        return self

    def predict(self, x):
        return self.design(x) @ self.coef


def make_model(seed=2027):
    return LinearExample(seed)


def synthetic_view(period, seed):
    rng = np.random.default_rng(seed)
    n = 48
    power, auxiliary = rng.uniform(0, 2, (2, n))
    known = np.arange(n) % 7 != 0
    x = np.c_[power, np.where(known, auxiliary, np.nan), known.astype(float), np.arange(n)//4 % 3]
    y = (1+2*power+3*np.where(known, auxiliary, 0))[:, None] * np.linspace(.8, 1.2, 10)[None, :]
    y += rng.normal(0, .03, y.shape)
    meta = pd.DataFrame(dict(source='Synthetic', site='Example', period=period,
        query_group=[f'q{i//4}' for i in range(n)], member=[f'd{i%4}' for i in range(n)],
        case_id=[f'{period}-{i}' for i in range(n)], date='2020-01-01'))
    # Every query has an input-only donor; this membership is retained.
    rows = np.flatnonzero(np.arange(n) % 4 != 3)
    donors = {s: donor_mapping(meta, 2, s)[0] for s in SEEDS}
    return RecordInputView(meta, x, rows, [1, 2], donors, period,
                           targets=y[rows] if period == 'train' else None), y[rows]


def run_example():
    train, _ = synthetic_view('train', 7)
    fits = []
    models = fit_record_models(train, make_model, progress=fits.append)
    rows = []
    for number, period in enumerate(['validation', 'later_1', 'later_2']):
        view, truth = synthetic_view(period, number+101)
        for method, prediction in predict_record_models(models, view, batch_size=13):
            error = prediction-truth
            rows.append(dict(period=period, method=method, cases=len(view),
                             mse=float(np.mean(error**2)), mae=float(np.mean(abs(error)))))
    return dict(purpose='Synthetic API usage demonstration, not benchmark evidence',
                fitted_models=len(fits), prediction_streams=len(rows), model_seed=2027,
                input_only_donors_per_split=12, results=rows, gpu_used=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='New JSON output; otherwise print only')
    args = parser.parse_args(argv)
    report = run_example()
    text = json.dumps(report, indent=2)+'\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x', encoding='utf-8') as stream:
            stream.write(text)
        print(json.dumps(dict(output=str(args.output), fitted_models=report['fitted_models'],
                              prediction_streams=report['prediction_streams'])))
    else:
        print(text)


if __name__ == '__main__':
    main()
