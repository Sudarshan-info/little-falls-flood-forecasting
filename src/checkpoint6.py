"""Audit the extended Little Falls record and the fixed calendar-year splits."""

import argparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config as C
from .qc import flood_event_ids
from .utils import CHECKPOINT6_CUTOFFS, checkpoint6_splits


HOUR = pd.Timedelta(hours=1)
QUARTER = pd.Timedelta(minutes=15)


def load_csv(path):
    return pd.read_csv(path, dtype={'time_series_id': str}, index_col='time_utc').rename_axis('time_utc')


def read_observations(path):
    frame = load_csv(path)
    frame.index = pd.to_datetime(frame.index, format='ISO8601', utc=True)
    return frame


def validate_raw(raw):
    start, _, _, stop = CHECKPOINT6_CUTOFFS
    if raw.index.hasnans or not raw.index.is_monotonic_increasing or not raw.index.is_unique:
        raise ValueError(f'Invalid raw timestamps or duplicates ({raw.index.duplicated().sum()}).')
    if (raw.index[0], raw.index[-1]) != (start, stop - QUARTER):
        raise ValueError('Raw coverage does not match the requested start and end.')
    if not raw.unit.eq('ft').all() or raw.time_series_id.isna().any() or raw.time_series_id.nunique() != 1:
        raise ValueError('Expected a single identified USGS time series with units of feet throughout.')
    if not np.isfinite(raw.stage_ft).all():
        raise ValueError('Nonfinite raw stage values require source review before preparation.')
    if not raw.approval_status.isin(['Approved', 'Provisional']).all():
        raise ValueError('Missing or unrecognized raw approval flags require source review.')
    # Broad triage for obvious unit errors or severe outliers; negative gage heights can be valid.
    if not raw.stage_ft.between(-2, 20).all():
        raise ValueError('Stage outside the broad -2 to 20 ft screening range; review before promotion.')


def gap_runs(missing, step):
    runs = []
    for _, series in missing.to_series().groupby(missing.to_series().diff().ne(step).cumsum()):
        runs.append({'first_missing_utc': series.iloc[0], 'last_missing_utc': series.iloc[-1],
                     'missing_slots': len(series)})
    return pd.DataFrame(runs, columns=['first_missing_utc', 'last_missing_utc', 'missing_slots'])


def summarize_events(study):
    above = study.stage_ft[study.stage_ft.ge(C.MINOR_STAGE)]
    ids = flood_event_ids(above.index, 24)
    rows = []
    for event, readings in above.groupby(ids):
        first, last = readings.index[[0, -1]]
        interval = study.stage_ft.loc[first:last]
        rows.append({
            'event_id': event, 'start_utc': first, 'end_utc': last, 'start_year': first.year,
            'duration_hours_inclusive': len(interval), 'flood_hours': len(readings),
            'below_threshold_hours_bridged': int(interval.lt(C.MINOR_STAGE).sum()),
            'unknown_hours_bridged': int(interval.isna().sum()),
            'peak_ft': readings.max(), 'peak_utc': readings.idxmax(),
        })
    return ids, pd.DataFrame(rows)


def add_gap_context(gaps, stage):
    for endpoint, direction, label in [('first', 'ffill', 'previous'), ('last', 'bfill', 'next')]:
        times = pd.DatetimeIndex(gaps[f'{endpoint}_missing_utc'])
        locations = stage.index.get_indexer(times, method=direction)
        gaps[f'{label}_observed_utc'] = stage.index[locations]
        gaps[f'{label}_stage_ft'] = stage.iloc[locations].to_numpy()
    return gaps


def plot_stage(hourly, folder):
    start, validation, test, stop = CHECKPOINT6_CUTOFFS
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), layout='constrained')
    for ax, first, last, title in [
        (axes[0], start, stop, 'Extended record | 2015–2025'),
        (axes[1], validation, stop, 'Holdout detail | Validation 2023 and test 2024–2025'),
    ]:
        stage = hourly.stage_ft.loc[first:last]
        ax.plot(stage.index, stage, color='#155a79', lw=.75, label='Hourly gage height')
        for level, color in zip((C.ACTION_STAGE, C.MINOR_STAGE, C.MODERATE_STAGE, C.MAJOR_STAGE),
                                ('#806600', '#ae5424', '#aa315b', '#643574')):
            ax.axhline(level, color=color, linestyle='--', lw=1, label=f'{level:g} ft')
        for cutoff in (validation, test):
            if first < cutoff < last:
                ax.axvline(cutoff, color='#4b5662', lw=1, alpha=.65)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.set(xlim=(first, last), ylabel='Stage (ft)', title=title)
        ax.grid(axis='y', alpha=.22)
    axes[0].legend(ncol=5, loc='lower left', framealpha=.95)
    axes[1].set_xlabel('Hourly cutoff (UTC)')
    fig.savefig(folder / 'stage_2015_2025.png', dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--staged', action='store_true', help='Audit the candidate files before promotion.')
    args = parser.parse_args()
    data = C.ROOT / 'data' / 'checkpoint6' if args.staged else C.ROOT / 'data'
    output = C.ROOT / 'outputs' / 'checkpoint6'
    output.mkdir(parents=True, exist_ok=True)
    raw = read_observations(data / 'raw' / 'measurements.csv')
    validate_raw(raw)
    hourly = read_observations(data / 'processed' / 'hourly_stage.csv')
    bounds = checkpoint6_splits(hourly.index)
    hourly['observed_time_utc'] = pd.to_datetime(hourly.observed_time_utc, format='ISO8601', utc=True)
    buckets = raw.assign(observed_time_utc=raw.index).resample('h', closed='right', label='right')
    rebuilt = buckets.agg(stage_ft=('stage_ft', 'last'), n_source_in_hour=('stage_ft', 'count'),
                          observed_time_utc=('observed_time_utc', 'last'))
    pd.testing.assert_frame_equal(hourly[rebuilt.columns], rebuilt, check_freq=False)
    available = hourly.observed_time_utc.notna()
    lag = hourly.index - hourly.observed_time_utc
    if not ((lag[available] >= pd.Timedelta(0)) & (lag[available] < HOUR)).all():
        raise ValueError('An hourly row uses a source timestamp outside its causal bucket.')

    start, _, _, stop = CHECKPOINT6_CUTOFFS
    grid = pd.date_range(start, stop, freq='15min', inclusive='left')
    missing = grid.difference(raw.index)
    off_grid = raw.index.difference(grid)
    study = hourly.loc[hourly.index < stop]
    missing_hours = study.index[study.stage_ft.isna()]
    hourly_gaps = add_gap_context(gap_runs(missing_hours, HOUR), study.stage_ft.dropna()) if len(missing_hours) else gap_runs(missing_hours, HOUR)
    if len(missing_hours) / len(study) > .02:
        raise ValueError('Over 2% of study hours lack an observation; investigate before promotion.')
    ids, events = summarize_events(study)
    ambiguous_separations = 0
    for last, first in zip(events.end_utc.iloc[:-1], events.start_utc.iloc[1:]):
        gap = study.stage_ft.loc[last + HOUR:first - HOUR]
        observed_below = gap.lt(C.MINOR_STAGE)
        if gap.isna().any() and not observed_below.rolling(24).sum().eq(24).any():
            ambiguous_separations += 1
    annual_rows = []
    for year, section in study.groupby(study.index.year):
        annual_rows.append({
            'year': year, 'hours': len(section), 'missing_hours': int(section.stage_ft.isna().sum()),
            'raw_missing_15min_slots': int((missing.year == year).sum()),
            'events_started': int(events.start_year.eq(year).sum()),
            'flood_hours': int(section.stage_ft.ge(C.MINOR_STAGE).sum()),
        })
    split_rows = []
    for name, (lo, hi) in bounds.items():
        section = study.iloc[lo:hi]
        event_hours = ids.reindex(section.index).dropna()
        split_rows.append({
            'split': 'validation' if name == 'val' else name,
            'first_cutoff_utc': section.index[0], 'last_cutoff_utc': section.index[-1],
            'hourly_rows': len(section), 'missing_hours': int(section.stage_ft.isna().sum()),
            'events_started': int(events.start_utc.between(section.index[0], section.index[-1]).sum()),
            'events_with_flood_hours': event_hours.nunique(), 'flood_hours': len(event_hours),
        })
    annual = pd.DataFrame(annual_rows)
    split_summary = pd.DataFrame(split_rows)
    if not (annual.flood_hours.sum() == split_summary.flood_hours.sum() == len(ids)):
        raise ValueError('Flood-hour totals disagree across the study, years and splits.')
    changes = raw.stage_ft.diff().abs().where(raw.index.to_series().diff().le(QUARTER))
    unusual = raw.loc[changes.nlargest(12).index].assign(abs_change_ft=changes.nlargest(12))
    extremes = raw.loc[raw.stage_ft.isin([raw.stage_ft.min(), raw.stage_ft.max()])]
    for filename, table in [('annual_flood_events', annual), ('chronological_split', split_summary),
                            ('flood_event_details', events), ('raw_gap_runs', gap_runs(missing, QUARTER)),
                            ('hourly_gap_runs', hourly_gaps), ('off_grid_readings', raw.loc[off_grid].reset_index()),
                            ('source_flags', raw.value_counts(['approval_status', 'qualifier'], dropna=False)
                             .rename('rows').reset_index()),
                            ('extremes', extremes.reset_index()), ('largest_short_changes', unusual.reset_index()),
                            ('hourly_boundaries', hourly.iloc[[0, -1]].reset_index())]:
        table.to_csv(output / f'{filename}.csv', index=False)
    plot_stage(hourly, output)
    print(f'Raw rows {len(raw)}, expected quarter-hour slots {len(grid)}, missing {len(missing)}, off-grid {len(off_grid)}.')
    print(f'Prepared rows {len(hourly)}, study hours {len(study)}, missing study hours {len(missing_hours)}.')
    print('Bucket sizes:', hourly.n_source_in_hour.value_counts().sort_index().to_dict())
    print('Source flags:', raw.value_counts(['approval_status', 'qualifier'], dropna=False).to_dict())
    print('Raw range:', raw.stage_ft.min(), raw.stage_ft.max())
    print('Largest short steps:', changes.nlargest(5).to_dict())
    print('Events bridging unknown hours:', int(events.unknown_hours_bridged.gt(0).sum()))
    print('Event separations without 24 confirmed below-threshold hours:', ambiguous_separations)
    print(annual.to_string(index=False))
    print(split_summary.to_string(index=False))


if __name__ == '__main__':
    main()
