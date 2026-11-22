"""Evaluate two fixed-lag stage forecasts on the chronological test split."""

import pandas as pd

from . import config as C
from .utils import complete_window_targets, forecast_splits, read_hourly, record


def main():
    stage = read_hourly().stage_ft
    start, stop = forecast_splits(stage.index)['test']
    actual = stage.iloc[start:stop]
    summary = []
    for name, lag in [('persistence', 1), ('seasonal_24h', 24)]:
        if start < lag:
            continue
        predicted = stage.shift(lag).where(complete_window_targets(stage, lag)).iloc[start:stop]
        scores = record(name, actual.index, actual.to_numpy(), predicted.to_numpy())
        summary.append({
            'model': name,
            'split': 'test',
            'lag_hours': lag,
            'actual_flood_hours': int(actual.ge(C.MINOR_STAGE).sum()),
            **scores,
        })
    pd.DataFrame(summary).to_csv(C.METRICS / 'qc_baseline_summary.csv', index=False)
    print('Stage forecast errors are separate from flood-warning performance.')


if __name__ == '__main__':
    main()
