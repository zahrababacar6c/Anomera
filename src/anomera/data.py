"""Data schema validation and chronological holdout construction."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

FEATURES = ['Time', *[f'V{i}' for i in range(1, 29)], 'Amount']
COLUMNS = FEATURES + ['Class']


def load_and_validate(path):
    df = pd.read_csv(path)
    if len(df) == 0:
        raise ValueError('Dataset is empty')
    if set(df.columns) != set(COLUMNS):
        raise ValueError(f'Unexpected columns. Missing: {set(COLUMNS)-set(df.columns)}; extra: {set(df.columns)-set(COLUMNS)}')
    df = df[COLUMNS]
    if df.isna().any().any():
        raise ValueError('Missing values detected; investigate before proceeding')
    if not all(pd.api.types.is_numeric_dtype(df[c]) for c in COLUMNS):
        raise ValueError('Non-numeric columns detected')
    if not np.isfinite(df.to_numpy(dtype=float)).all():
        raise ValueError('Non-finite values detected')
    if not df['Class'].isin([0, 1]).all():
        raise ValueError('Class must contain only 0 and 1')
    if (df['Time'] < 0).any() or (df['Amount'] < 0).any():
        raise ValueError('Time and Amount must be nonnegative')
    return df


def chronological_split(df, train_fraction=0.60, validation_fraction=0.20):
    """Stable, time-ordered 60/20/20 split; no fitting or resampling here."""
    if not (0 < train_fraction < 1 and 0 < validation_fraction < 1 and train_fraction + validation_fraction < 1):
        raise ValueError('Invalid split fractions')
    ordered = df.sort_values('Time', kind='mergesort').reset_index(drop=True)
    n = len(ordered)
    a, b = int(n * train_fraction), int(n * (train_fraction + validation_fraction))
    if not (0 < a < b < n):
        raise ValueError('Insufficient rows for split')
    return ordered.iloc[:a].copy(), ordered.iloc[a:b].copy(), ordered.iloc[b:].copy()


def create_splits(source, output_dir):
    df = load_and_validate(source)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    train, val, test = chronological_split(df)
    manifest = {'source': str(source), 'split_method': 'chronological_stable_60_20_20', 'dataset_rows': len(df), 'dataset_duplicates': int(df.duplicated().sum()), 'splits': {}}
    for name, part in [('train',train), ('validation',val), ('test',test)]:
        part.to_csv(out / f'{name}.csv', index=False)
        manifest['splits'][name] = {'rows': len(part), 'frauds': int(part['Class'].sum()), 'fraud_rate': float(part['Class'].mean()), 'time_min': float(part['Time'].min()), 'time_max': float(part['Time'].max())}
    (out / 'split_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return manifest
