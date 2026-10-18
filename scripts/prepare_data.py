"""Run from repository root: python scripts/prepare_data.py"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from anomera.data import create_splits

if __name__ == '__main__':
    import json
    print(json.dumps(create_splits(ROOT / 'data/raw/creditcard.csv', ROOT / 'data/processed'), indent=2))
