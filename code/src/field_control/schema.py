"""Public field semantics for model authors; no data-dependent recoding."""
from copy import deepcopy

_SCHEMA = {
    'version': 'field-control-inputs-1.1',
    'state_order': ['power_kw', 'present', 'capacity', 'oper_state', 'running', 'opt_out'],
    'history_state_order': ['present', 'capacity', 'oper_state', 'running', 'opt_out'],
    'fields': {
        'power_kw': {'kind': 'continuous', 'unit': 'kW',
            'meaning': 'Provider-reported, partly estimated/calibrated device power.'},
        'present': {'kind': 'continuous', 'unit': 'Wh',
            'source_name': 'present_whr', 'meaning': 'Native reported energy-state quantity; zero is not imputed missingness.'},
        'capacity': {'kind': 'continuous', 'unit': 'Wh',
            'source_name': 'total_whr', 'meaning': 'Native reported total energy quantity, not a device power rating.'},
        'oper_state': {'kind': 'categorical', 'unit': None,
            'standard_codes': list(range(13)), 'unused_range': [13,125],
            'manufacturer_reserved_range': [126,255],
            'meaning': 'Operating-state code. Numeric distance between codes has no physical interpretation.',
            'missingness': 'Use the observation mask. Reserved code255 is not automatically missing.'},
        'running': {'kind': 'categorical', 'unit': None, 'binary_codes': [0,1],
            'other_codes': 'Preserve as other observed categories; do not treat255 as a large running magnitude.'},
        'opt_out': {'kind': 'categorical', 'unit': None, 'binary_codes': [0,1],
            'other_codes': 'Preserve observed codes and use observation masks for unavailable reports.'},
        'command_type': {'kind': 'categorical', 'unit': None,
            'meaning': 'Observed control-code namespace supplied by the field logger.'},
        'command_numeric': {'kind': 'raw_numeric_parameter', 'unit': None,
            'meaning': 'Retain with its associated command type; do not assume it represents minutes.'},
        'observation_mask': {'kind': 'indicator', 'unit': None,
            'recommended_numeric_range': [0,1]},
        'observed_fraction': {'kind': 'coverage_fraction', 'unit': None,
            'recommended_numeric_range': [0,1]}
    },
    'availability': {
        'current': 'First online report carrying a changed observed command; not verified dispatch time.',
        'history': 'Reports strictly before the origin, with explicit observation masks.',
        'feedback': 'Only reports up to the stated updated query time.',
        'future_commands': 'Not inputs in the current-command track; no complete as-issued future plan ledger is available.'
    },
    'targets': {'unit': 'kW', 'meaning': 'Future reported power, not counterfactual energy savings.'},
    'reference_encoding': 'Typed one-hot categorical fields; scale continuous quantities only. Masks/indicators stay bounded.',
    'source': {'operating_states': 'BPA Demand Response Market Transformation and Business Case Report, printed p156 / physical PDF p171.'}
}


def input_schema():
    """Return an independent schema dictionary for data consumers."""
    return deepcopy(_SCHEMA)
