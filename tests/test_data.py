import sys
from pathlib import Path
import pandas as pd
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from anomera.data import FEATURES, chronological_split, load_and_validate


def sample():
    d = {c: [float(i) for i in range(10)] for c in FEATURES}
    d['Time'] = [float(i) for i in range(10)]
    d['Class'] = [0,0,1,0,0,0,1,0,0,1]
    return pd.DataFrame(d)


def test_split_is_chronological_and_disjoint():
    df = sample().sample(frac=1, random_state=1)
    a,b,c = chronological_split(df)
    assert (len(a),len(b),len(c)) == (6,2,2)
    assert a.Time.max() <= b.Time.min() <= c.Time.min()
    assert set(a.Time).isdisjoint(b.Time) and set(b.Time).isdisjoint(c.Time)


def test_schema_validation(tmp_path):
    p = tmp_path / 'creditcard.csv'
    sample().to_csv(p, index=False)
    assert len(load_and_validate(p)) == 10
    bad = sample().drop(columns=['V1'])
    bad.to_csv(p, index=False)
    with pytest.raises(ValueError, match='Unexpected columns'):
        load_and_validate(p)
