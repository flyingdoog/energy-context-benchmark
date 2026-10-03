"""Model-neutral training and prediction for the record-association comparison."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .record_linkage import reassign_blocks
from .record_reference_inputs import SEEDS


class RecordInputView:
    """Labeled recipients with a separate full query-known donor population.

    The tabular encoding is the portable BPA/ACN reference encoding. It is not
    an automatically reshaped sequence interface. Missing values and their
    existing mask columns are retained.
    """
    def __init__(self, metadata, features, recipients, moved_columns, donors,
                 period, targets=None, output_width=10):
        self.metadata = metadata.reset_index(drop=True).copy()
        self._features = np.asarray(features)
        self.recipients = np.asarray(recipients, dtype=int)
        self.moved_columns = np.asarray(moved_columns, dtype=int)
        self.donors = {int(k): np.asarray(v, dtype=int) for k, v in donors.items()}
        self.period, self.output_width = period, int(output_width)
        self._targets = None if targets is None else np.asarray(targets)
        if self._features.ndim != 2 or len(self.metadata) != len(self._features):
            raise ValueError('Metadata and full query-known input population differ')
        if not self.metadata.case_id.is_unique:
            raise ValueError('Case identities must be unique')
        if self.recipients.ndim != 1 or len(np.unique(self.recipients)) != len(self.recipients):
            raise ValueError('Recipients must be unique rows')
        if ((self.recipients < 0) | (self.recipients >= len(self.metadata))).any():
            raise ValueError('Recipient outside query-known population')
        if set(self.donors) != set(SEEDS):
            raise ValueError('All three original mappings are required')
        # Validate dimensions without materializing the full feature matrix.
        for donor in self.donors.values():
            reassign_blocks(self._features, donor, self.moved_columns, self.recipients[:0])
        if self.period != 'train' and targets is not None:
            raise ValueError('Evaluation views do not expose labels to model callbacks')
        if self.period == 'train' and (self._targets is None or
                self._targets.shape != (len(self.recipients), self.output_width) or
                not np.isfinite(self._targets).all()):
            raise ValueError('Training targets must match labeled recipient rows')

    @classmethod
    def from_cache(cls, cache, period):
        folder = Path(cache)/period
        if not (folder/'complete.json').is_file():
            raise ValueError('Prepare the complete input period first')
        metadata = pd.read_csv(folder/'metadata.csv.gz', dtype={'case_id': str, 'member': str})
        features = np.load(folder/'features.npy', mmap_mode='r', allow_pickle=False)
        with np.load(folder/'labeled.npz', allow_pickle=False) as packet:
            recipients, moved = packet['rows'], packet['moved']
            targets = packet['target_kw'] if period == 'train' else None
        donors = {seed: np.load(folder/f'donor_{seed}.npy', allow_pickle=False) for seed in SEEDS}
        return cls(metadata, features, recipients, moved, donors, period, targets)

    @property
    def case_ids(self):
        return self.metadata.case_id.iloc[self.recipients].to_numpy(dtype=str)

    def __len__(self):
        return len(self.recipients)

    def training_targets(self):
        if self.period != 'train':
            raise ValueError('Only the training view exposes training targets')
        return self._targets.copy()

    def features(self, arm='paired', mapping_seed=None, start=0, stop=None):
        stop = len(self) if stop is None else stop
        if not 0 <= start <= stop <= len(self):
            raise ValueError('Invalid recipient batch')
        rows = self.recipients[start:stop]
        if arm == 'reassigned':
            if mapping_seed not in SEEDS:
                raise ValueError('Reassigned inputs require an original mapping seed')
            return reassign_blocks(self._features, self.donors[mapping_seed], self.moved_columns, rows)
        if arm not in ('power', 'paired') or mapping_seed is not None:
            raise ValueError('Choose power, paired, or reassigned with a mapping seed')
        values = self._features[rows].copy()
        if arm == 'power':
            values[:, self.moved_columns] = 0
        return values


def fit_record_models(train, model_factory, model_seed=2027, progress=None):
    """Factory(seed=...) -> fresh object with fit(X, Y) and predict(X).

    Exactly five objects are fitted. All receive the same model seed;
    reassignment seeds change inputs, not model initialization.
    """
    if train.period != 'train':
        raise ValueError('Model fitting requires the training view')
    models = {}
    specs = [('power', 'power', None), ('paired', 'paired', None)]
    specs += [(f'retrained_{seed}', 'reassigned', seed) for seed in SEEDS]
    for name, arm, seed in specs:
        model = model_factory(seed=model_seed)
        if any(model is previous for previous in models.values()):
            raise ValueError('The factory must return a fresh model for each training arm')
        if not callable(getattr(model, 'fit', None)) or not callable(getattr(model, 'predict', None)):
            raise TypeError('Models need callable fit(X, Y) and predict(X)')
        model.fit(train.features(arm, seed), train.training_targets())
        models[name] = model
        if progress:
            progress(dict(stage='fit_model', arm=name, training_cases=len(train)))
    return models


def predict_record_models(models, view, batch_size=4096, clip_nonnegative=False):
    """Yield eight complete-split (method, prediction) packets without fitting.

    Paired and all three fixed-reassignment streams use exactly the same
    paired object. Predict must not adapt or update its fitted state.
    """
    if batch_size < 1:
        raise ValueError('Prediction batch size must be positive')
    required = {'power', 'paired'} | {f'retrained_{seed}' for seed in SEEDS}
    if set(models) != required:
        raise ValueError('Expected the five fitted record-comparison models')
    specs = [('Power only', 'power', 'power', None), ('Paired', 'paired', 'paired', None)]
    specs += [(f'Fixed {s}', 'paired', 'reassigned', s) for s in SEEDS]
    specs += [(f'Retrained {s}', f'retrained_{s}', 'reassigned', s) for s in SEEDS]
    for method, name, arm, seed in specs:
        prediction = np.empty((len(view), view.output_width), dtype=np.float32)
        for start in range(0, len(view), batch_size):
            stop = min(start+batch_size, len(view))
            values = np.asarray(models[name].predict(view.features(arm, seed, start, stop)))
            if values.shape != (stop-start, view.output_width) or not np.isfinite(values).all():
                raise ValueError('Predict must return finite (batch cases, 10) power values in kW')
            prediction[start:stop] = np.maximum(values, 0) if clip_nonnegative else values
        if not np.isfinite(prediction).all():
            raise ValueError('Predictions exceed portable float32 range')
        yield method, prediction


def load_record_cache(cache, data, source, site=None):
    """Read a complete cache built by the source-based comparison preparer."""
    cache, data = Path(cache).resolve(), Path(data).resolve()
    plan = json.loads((cache/'plan.json').read_text(encoding='utf-8'))
    receipt = json.loads((cache/'complete.json').read_text(encoding='utf-8'))
    if (Path(plan['data_root']).resolve() != data or plan['source'] != source or
            plan['site'] != site or plan['input_recipe'] != 'record-reference-v1'):
        raise ValueError('Prepared input cache belongs to a different data root or task')
    if not receipt.get('completed'):
        raise ValueError('Prepared input cache is incomplete')
    expected = ['train', 'validation'] + (['spring', 'summer'] if source == 'BPA' else ['autumn', 'winter'])
    if [item['period'] for item in receipt['support']] != expected:
        raise ValueError('The cache must retain all original named splits')
    return receipt
