"""Verify cached USGS observations and stage the canonical Little Falls dataset."""

import json
from hashlib import file_digest

import numpy as np
import pandas as pd

from . import config as C
from .checkpoint6 import HOUR, QUARTER, gap_runs, read_observations
from .fetch import FIELDS, months
from .historical_extension import classify_events
from .prepare import canonical_hourly_stage
from .utils import CANONICAL_CUTOFFS, complete_window_targets, forecast_splits


SERIES = '55e58219c537444982291c6127f7cb68'
START, _, _, STOP = CANONICAL_CUTOFFS
SEAM = pd.Timestamp('2015-01-01', tz='UTC')
SERVICE_SEAM = pd.Timestamp('2010-05-01', tz='UTC')
HISTORY = C.ROOT / 'data' / 'exploration' / '2007_2014'
BACKUP = C.ROOT / 'data' / 'backups' / 'pre-canonical-2015-2025'
STAGING = C.ROOT / 'data' / 'canonical_stage'
REPORT = C.ROOT / 'outputs' / 'canonical_qc'


def sha256(path):
    with path.open('rb') as source:
        return file_digest(source, 'sha256').hexdigest()


class DuplicateTimestamps(ValueError):
    def __init__(self, rows):
        self.rows = rows
        super().__init__(f'{rows.index.nunique()} duplicate timestamps require review: '
                         f'{rows.duplicate_type.value_counts().to_dict()} rows')


def merge_observations(historical, modern):
    combined = pd.concat([historical.assign(source_snapshot='2007-2014'),
                          modern.assign(source_snapshot='2015-2025')]).sort_index(kind='stable')
    duplicate = combined.loc[combined.index.duplicated(keep=False)].copy()
    if not duplicate.empty:
        fields = duplicate.drop(columns='source_snapshot')
        kinds = fields.groupby(level=0).nunique(dropna=False).gt(1).any(axis=1)
        duplicate['duplicate_type'] = np.where(duplicate.index.map(kinds), 'conflicting', 'identical')
        raise DuplicateTimestamps(duplicate)
    return combined.drop(columns='source_snapshot')


def validate_source(raw, first, last):
    if list(raw.columns) != list(FIELDS)[1:] or raw.index.hasnans or str(raw.index.tz) != 'UTC':
        raise ValueError('Unexpected CSV schema or invalid UTC timestamps.')
    if not raw.index.is_monotonic_increasing or (raw.index[0], raw.index[-1]) != (first, last):
        raise ValueError('Source timestamps are unordered or do not match known coverage.')
    if raw[['unit', 'time_series_id']].isna().any().any() or not (
        raw.unit.eq('ft').all() and raw.time_series_id.eq(SERIES).all()
    ):
        raise ValueError('Unit or USGS primary series changed; investigate before merging.')
    if raw.approval_status.isna().any() or not raw.approval_status.eq('Approved').all() or raw.qualifier.notna().any():
        raise ValueError('Unexpected approval or qualifier flags; review the source before merging.')
    if not np.isfinite(raw.stage_ft).all() or not raw.stage_ft.between(-2, 20).all():
        raise ValueError('Unavailable or outlying stage value; review the source.')


def monthly_cache(folder, start, end):
    frames = []
    for begin, finish, _ in months(start, end):
        path = folder / f'usgs_{C.SITE}_{C.PARAMETER}_{begin}_{finish}.csv'
        frames.append(read_observations(path))
    return pd.concat(frames)


def legacy_cache():
    frames = []
    for year in range(2010, 2015):
        body = json.loads((HISTORY / f'usgs_legacy_iv_{year}.json').read_bytes())
        (series,) = body['value']['timeSeries']
        (site,) = series['sourceInfo']['siteCode']
        variable = series['variable']
        if (site['agencyCode'], site['value'], variable['variableCode'][0]['value'],
                variable['unit']['unitCode'], series['name']) != (
                    'USGS', '01389500', '00065', 'ft', 'USGS:01389500:00065:00000'):
            raise ValueError('Legacy source is not the expected USGS site, parameter and unit.')
        rows = [row for group in series['values'] for row in group['value']]
        if any(row['qualifiers'] != ['A'] or float(row['value']) == variable['noDataValue'] for row in rows):
            raise ValueError('Unexpected legacy flag or missing-value sentinel.')
        times = pd.to_datetime([row['dateTime'] for row in rows], format='ISO8601', utc=True)
        frame = pd.DataFrame({'stage_ft': [float(row['value']) for row in rows], 'unit': 'ft',
                              'approval_status': 'Approved', 'qualifier': np.nan, 'time_series_id': SERIES},
                             index=times.rename('time_utc')).sort_index()
        frames.append(frame.loc[frame.index.year == year])
    return pd.concat(frames)


def confirm_provenance(historical, modern):
    historical_ogc = monthly_cache(HISTORY, '2007-10-01', '2010-04-30')
    historical_iv = legacy_cache()
    overlap = historical_ogc.loc[historical_ogc.index >= historical_iv.index[0]]
    pd.testing.assert_frame_equal(overlap, historical_iv.loc[
        (historical_iv.index >= historical_iv.index[0]) & (historical_iv.index < SERVICE_SEAM)])
    pd.testing.assert_frame_equal(historical, pd.concat([
        historical_ogc, historical_iv.loc[historical_iv.index >= SERVICE_SEAM]]))
    pd.testing.assert_frame_equal(modern, monthly_cache(C.RAW, '2015-01-01', '2025-12-31'))
    archived = read_observations(C.ROOT / 'data' / 'backups' / '2021-2025' / 'measurements.csv')
    pd.testing.assert_frame_equal(archived, modern.loc[archived.index[0]:archived.index[-1]])
    return {'legacy_ogc_overlap_rows': len(overlap), 'archived_2021_2025_overlap_rows': len(archived),
            'historical_ogc_rows': len(historical_ogc),
            'historical_legacy_rows': int((historical_iv.index >= SERVICE_SEAM).sum())}


def summary_row(label, stage, events):
    known = stage.dropna()
    gaps = gap_runs(stage.index[stage.isna()], HOUR)
    started = events.start_utc.between(stage.index[0], stage.index[-1])
    overlaps = events.start_utc.le(stage.index[-1]) & events.end_utc.ge(stage.index[0])
    return {'period': label, 'first_cutoff_utc': stage.index[0], 'last_cutoff_utc': stage.index[-1],
            'study_hours': len(stage), 'valid_hours': len(known), 'missing_hours': len(stage) - len(known),
            'longest_missing_run_hours_within_period': int(gaps.missing_slots.max()) if len(gaps) else 0,
            'min_stage_ft': known.min(), 'max_stage_ft': known.max(),
            'hours_ge_action_6_5_ft': int(known.ge(C.ACTION_STAGE).sum()),
            'flood_hours_ge_7_ft': int(known.ge(C.MINOR_STAGE).sum()),
            'flood_events_started': int(started.sum()), 'flood_events_present': int(overlaps.sum())}


def plot_qc_summary(annual, split, destination):
    """Draw aggregate counts suitable for sharing separately from the time series."""
    import matplotlib.pyplot as plt

    fig, (top, bottom) = plt.subplots(2, 1, figsize=(11, 7), layout='constrained')
    years = annual.period.astype(int)
    top.bar(years, annual.flood_hours_ge_7_ft, color='#1e6381')
    top.set(title='Annual flood-stage hours | USGS 01389500', ylabel='Hours at or above 7.0 ft')
    top.set_xticks(years[::2])
    bars = bottom.bar(split.period.str.capitalize(), split.flood_events_started, color='#a94534')
    bottom.bar_label(bars, padding=3)
    bottom.set(title='Independent flood events by fixed split', ylabel='Flood events',
               ylim=(0, max(1, split.flood_events_started.max() * 1.22)))
    for ax in (top, bottom):
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', alpha=.22)
        ax.set_axisbelow(True)
    fig.savefig(destination, dpi=160)
    plt.close(fig)


def stage_and_audit(raw, provenance):
    hourly = canonical_hourly_stage(raw)
    bounds = forecast_splits(hourly.index)
    known = hourly.observed_time_utc.notna()
    source = hourly.loc[known, 'observed_time_utc']
    offsets = source.index - source
    if not offsets.between(pd.Timedelta(0), HOUR, inclusive='left').all():
        raise ValueError('An hourly row includes an observation after its cutoff or before its interval.')
    study = hourly.iloc[:bounds['test'][1]].stage_ft
    events = classify_events(study)
    if events.empty or not events.status.eq('confirmed').all():
        raise ValueError('Flood episode boundaries or separation are uncertain. Inspect before promotion.')
    periods = {'train': 'train', 'val': 'validation', 'test': 'test'}
    split = pd.DataFrame([summary_row(periods[name], study.iloc[lo:hi], events)
                          for name, (lo, hi) in bounds.items()])
    annual_rows = []
    for year, values in study.groupby(study.index.year):
        row = summary_row(year, values, events)
        row['valid_raw_measurements'] = int((raw.index.year == year).sum())
        annual_rows.append(row)
    annual = pd.DataFrame(annual_rows)
    if split.flood_events_started.tolist() != [14, 3, 3]:
        raise ValueError(f'Independent event counts differ from the expected audit: {split.flood_events_started.tolist()}')
    if not split.flood_events_started.equals(split.flood_events_present):
        raise ValueError('An event spans split boundaries. Review the split and event inventory.')
    if not (annual.flood_hours_ge_7_ft.sum() == split.flood_hours_ge_7_ft.sum() == events.hours_ge_7_ft.sum()):
        raise ValueError('Annual, split, and event flood-hour totals disagree.')

    raw_grid = pd.date_range(START, STOP, freq='15min', inclusive='left')
    missing_raw = raw_grid.difference(raw.index)
    off_grid = raw.index.difference(raw_grid)
    gaps = gap_runs(study.index[study.isna()], HOUR)
    preceding = gaps.first_missing_utc - HOUR
    following = gaps.last_missing_utc + HOUR
    gaps['previous_hour_stage_ft'] = study.reindex(preceding).to_numpy()
    gaps['next_hour_stage_ft'] = study.reindex(following).to_numpy()
    windows = []
    for lookback in (1, 4, 12, 24, 48, 72):
        eligible = complete_window_targets(study, lookback)
        for name, (lo, hi) in bounds.items():
            windows.append({'split': periods[name], 'lookback_hours': lookback,
                            'eligible_targets': int(eligible[lo:hi].sum()), 'total_targets': hi - lo})

    paths = {'raw': STAGING / 'measurements.csv', 'hourly': STAGING / 'hourly_stage.csv'}
    STAGING.mkdir(parents=True, exist_ok=True)
    raw.to_csv(paths['raw'])
    hourly.to_csv(paths['hourly'])
    for name, table in [('annual_statistics', annual), ('chronological_split', split),
                        ('flood_events', events), ('hourly_gap_runs', gaps),
                        ('raw_gap_runs', gap_runs(missing_raw, QUARTER)),
                        ('off_grid_readings', raw.loc[off_grid].reset_index()),
                        ('window_eligibility', pd.DataFrame(windows))]:
        table.to_csv(REPORT / f'{name}.csv', index=False)
    plot_qc_summary(annual, split, REPORT / 'flood_events_and_hours.png')
    report = {
        'station': C.SITE, 'parameter': C.PARAMETER, 'unit': 'ft', 'primary_series_id': SERIES,
        'flood_rule': 'Stage >= 7.0 ft. Separate events with 24 consecutive observed hours below 7.0 ft.',
        'provenance': provenance, 'raw_rows': len(raw), 'raw_first_utc': str(raw.index[0]),
        'raw_last_utc': str(raw.index[-1]), 'raw_duplicates': 0, 'raw_approved_rows': len(raw),
        'raw_qualified_rows': 0, 'raw_missing_quarter_hours': len(missing_raw), 'raw_off_grid': len(off_grid),
        'raw_min_ft': raw.stage_ft.min(), 'raw_min_first_utc': str(raw.stage_ft.idxmin()),
        'raw_max_ft': raw.stage_ft.max(), 'raw_max_first_utc': str(raw.stage_ft.idxmax()),
        'hourly_rows': len(hourly), 'hourly_first_utc': str(hourly.index[0]),
        'hourly_last_utc': str(hourly.index[-1]), 'study_hours': len(study),
        'valid_study_hours': int(study.notna().sum()), 'missing_study_hours': int(study.isna().sum()),
        'hourly_gap_runs': len(gaps), 'longest_hourly_gap': gaps.loc[gaps.missing_slots.idxmax()].to_dict(),
        'hourly_min_ft': study.min(), 'hourly_max_ft': study.max(),
        'hourly_bucket_sizes': hourly.n_source_in_hour.value_counts().sort_index().to_dict(),
        'closing_bucket_source_rows': int(hourly.n_source_in_hour.iloc[-1]),
        'closing_bucket_in_study': False, 'events': len(events),
        'flood_hours': int(events.hours_ge_7_ft.sum()),
        'sha256': {name: sha256(path) for name, path in paths.items()},
        'bytes': {name: path.stat().st_size for name, path in paths.items()},
    }
    (REPORT / 'dataset_qc.json').write_text(json.dumps(report, indent=2, default=str) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2, default=str))
    print(annual[['period', 'missing_hours', 'flood_events_started', 'flood_hours_ge_7_ft']].to_string(index=False))
    print(split.to_string(index=False))


def main():
    REPORT.mkdir(parents=True, exist_ok=True)
    historical = read_observations(HISTORY / 'measurements.csv')
    modern = read_observations(BACKUP / 'measurements.csv')
    validate_source(historical, START + 5 * HOUR, SEAM - QUARTER)
    validate_source(modern, SEAM, STOP - QUARTER)
    try:
        raw = merge_observations(historical, modern)
    except DuplicateTimestamps as error:
        error.rows.to_csv(REPORT / 'duplicate_timestamp_review.csv')
        raise
    provenance = confirm_provenance(historical, modern)
    provenance.update({
        'historical_snapshot_sha256': sha256(HISTORY / 'measurements.csv'),
        'modern_snapshot_sha256': sha256(BACKUP / 'measurements.csv'),
        'cross_snapshot_overlap_rows': 0, 'seam_separation_minutes': int(
            (modern.index[0] - historical.index[-1]) / pd.Timedelta(minutes=1)),
        'legacy_interval_source': 'https://waterservices.usgs.gov/nwis/iv/',
        'modern_source': 'https://api.waterdata.usgs.gov/ogcapi/v1/collections/continuous/items',
        'historical_gage_reference': 'USGS Irene report uses 7.0 ft and 14.19 ft crest at station 01389500.',
        'threshold_and_datum_source': 'https://water.noaa.gov/gauges/01389500',
        'historical_threshold_source': 'https://pubs.usgs.gov/sir/2013/5234/',
        'datum_offset_applied': False,
    })
    stage_and_audit(raw, provenance)


if __name__ == '__main__':
    main()
