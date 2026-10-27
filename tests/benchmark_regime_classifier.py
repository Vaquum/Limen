"""Characterize fixed regime hypotheses on recorded periods without fitting them."""
import argparse
from dataclasses import asdict, replace
import hashlib
import json
import logging
from pathlib import Path
import platform
import time

import polars as pl

from limen.features.classify_market_regime import RegimeConfig, classify_market_regime

LOGGER = logging.getLogger(__name__)
INPUTS = ('spot_1h_20240101_20241231.parquet', 'spot_15m_20250101_20250531.parquet')


def configurations():
    baseline = RegimeConfig(version='baseline-v1')
    return {'baseline': baseline,
            'trend_entry_060': replace(baseline, trend_rank_entry=.60),
            'trend_entry_070': replace(baseline, trend_rank_entry=.70),
            'breakout_atr_020': replace(baseline, breakout_atr=.20),
            'breakout_atr_030': replace(baseline, breakout_atr=.30),
            'confirmation_1': replace(baseline, confirmation_rows=1),
            'confirmation_3': replace(baseline, confirmation_rows=3),
            'breakout_ttl_2': replace(baseline, breakout_ttl=2),
            'breakout_ttl_4': replace(baseline, breakout_ttl=4)}


def characterize(root, output):
    observations = []
    for name in INPUTS:
        path = root / 'tests/fixtures' / name
        bars = pl.read_parquet(path).with_columns(pl.lit(path.stem).alias('series_id'))
        baseline = None
        for identity, config in configurations().items():
            started = time.perf_counter()
            result = classify_market_regime(bars, config=config)
            codes = result['market_regime_code']
            if baseline is None:
                baseline = codes
            counts = result.group_by('market_regime').len().sort('market_regime')
            stress = result['is_stress'].drop_nulls()
            record = {'input': str(path.relative_to(root)), 'input_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                      'first_datetime': bars['datetime'][0].isoformat(), 'last_datetime': bars['datetime'][-1].isoformat(),
                      'rows': result.height, 'configuration': identity, 'config': asdict(config),
                      'config_hash': config.config_hash, 'feature_hash': result['regime_feature_hash'][0],
                      'regime_counts': dict(zip(counts['market_regime'].to_list(), counts['len'].to_list(), strict=True)),
                      'committed_label_changes': (codes != codes.shift(1)).sum(),
                      'labels_differing_from_baseline': (codes != baseline).sum(),
                      'stress_rows': stress.sum(), 'stress_available_rows': stress.len()}
            observations.append(record)
            LOGGER.info('%s %s %s %.3fs %s', name, identity, result.height,
                        time.perf_counter() - started, record['regime_counts'])
    report = {'scope': 'Chronological descriptive characterization of fixed hypotheses; no optimization, predictive accuracy or trading-return claim.',
              'python': platform.python_version(), 'platform': platform.system(), 'machine': platform.machine(),
              'observations': observations}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + '\n')
    return report


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    characterize(Path(__file__).resolve().parents[1], args.output)
