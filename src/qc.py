"""Reproduce the read-only audit of the 2021–2025 Little Falls study data."""

import hashlib

import matplotlib
matplotlib.use('Agg')
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config as C
from .utils import splits


def flood_event_ids(index, separation_hours):
    # A 25-hour timestamp gap has 24 intervening below-threshold rows.
    breaks = index.to_series().diff().ge(pd.Timedelta(hours=separation_hours + 1))
    return breaks.cumsum().add(1).rename('event_id')


def main():
    raw_path = C.RAW / 'measurements.csv'
    hourly_path = C.PROCESSED / 'hourly_stage.csv'
    original = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in (raw_path, hourly_path)}
    raw = pd.read_csv(raw_path, dtype={'time_series_id': str})
    hourly = pd.read_csv(hourly_path)
    for frame in (raw, hourly):
        frame['time_utc'] = pd.to_datetime(frame['time_utc'], format='ISO8601', utc=True)
        assert frame.time_utc.is_unique and frame.time_utc.is_monotonic_increasing
        assert np.isfinite(frame.stage_ft).all()
    hourly['observed_time_utc'] = pd.to_datetime(hourly['observed_time_utc'], format='ISO8601', utc=True)
    raw = raw.set_index('time_utc')
    hourly = hourly.set_index('time_utc')
    start = pd.Timestamp('2021-01-01', tz='UTC')
    stop = pd.Timestamp('2026-01-01', tz='UTC')
    hour = pd.Timedelta(hours=1)
    quarter = pd.Timedelta(minutes=15)
    if raw.index.min() != start or raw.index.max() != stop - quarter:
        raise SystemExit('This audit expects the 2021–2025 study snapshot. Check the input files.')
    assert raw.unit.eq('ft').all() and raw.time_series_id.nunique() == 1
    assert hourly.index.equals(pd.date_range(start, stop, freq='h', name='time_utc'))
    expected = pd.date_range(start, stop, freq='15min', inclusive='left')
    gaps = expected.difference(raw.index)
    off_grid = raw.index.difference(expected)
    buckets = raw.assign(observed_time_utc=raw.index).resample('h', closed='right', label='right')
    rebuilt = buckets.agg(stage_ft=('stage_ft', 'last'), n_source_in_hour=('stage_ft', 'count'),
                          observed_time_utc=('observed_time_utc', 'last'))
    pd.testing.assert_frame_equal(hourly[rebuilt.columns], rebuilt, check_freq=False)
    offsets = hourly.index - hourly.observed_time_utc
    assert (offsets >= pd.Timedelta(0)).all() and (offsets < hour).all()
    forecast_future = (hourly.observed_time_utc.iloc[1:].array > hourly.index[:-1])
    assert forecast_future.all()

    metrics = C.METRICS
    figures = C.FIGURES
    thresholds = (C.ACTION_STAGE, C.MINOR_STAGE, C.MODERATE_STAGE, C.MAJOR_STAGE)
    years = range(2021, 2026)
    in_period = hourly.loc[(hourly.index >= start) & (hourly.index < stop)]
    flood = hourly.loc[hourly.stage_ft.ge(C.MINOR_STAGE), 'stage_ft']
    event_ids = flood_event_ids(flood.index, 24)
    events = []
    for eid, readings in flood.groupby(event_ids):
        first, last = readings.index[[0, -1]]
        elapsed = int((last - first) / hour)
        events.append(dict(event_id=eid, start_utc=first, end_utc=last, start_year=first.year,
                           elapsed_hours=elapsed, duration_hours_inclusive=elapsed+1,
                           flood_hours=len(readings), below_threshold_hours_bridged=elapsed+1-len(readings),
                           peak_stage_ft=readings.max(), peak_time_utc=readings.idxmax()))
    events = pd.DataFrame(events)

    annual = []
    for year in years:
        stages = in_period.loc[in_period.index.year == year, 'stage_ft']
        result = dict(year=year, hourly_rows=len(stages), events_started=int(events.start_year.eq(year).sum()))
        for threshold in thresholds:
            label = str(threshold).replace('.', '_')
            result[f'hours_ge_{label}_ft'] = int(stages.ge(threshold).sum())
            result[f'hours_gt_{label}_ft'] = int(stages.gt(threshold).sum())
        annual.append(result)
    annual = pd.DataFrame(annual)

    n = len(hourly)
    bounds = splits(n)
    split_rows = []
    split_by_event = []
    for name, (lo, hi) in bounds.items():
        section = hourly.iloc[lo:hi]
        ids = event_ids.reindex(section.index).dropna().astype(int)
        started = events.start_utc.between(section.index[0], section.index[-1]).sum()
        split_rows.append(dict(split={'val': 'validation'}.get(name, name), start_utc=section.index[0], end_utc=section.index[-1],
                           hourly_rows=len(section), flood_hours=len(ids),
                           events_with_flood_hours=ids.nunique(), events_started=int(started)))
        for eid, count in ids.value_counts(sort=False).items():
            split_by_event.append(dict(event_id=eid, split={'val': 'validation'}.get(name, name), flood_hours=count))
    split_summary = pd.DataFrame(split_rows)
    split_by_event = pd.DataFrame(split_by_event)

    sensitivity_rows = []
    for separation in (1, 6, 12, 24, 48, 72, 168):
        ids = flood_event_ids(flood.index, separation)
        row = {'minimum_below_threshold_hours': separation, 'events': ids.nunique(), 'flood_hours': len(ids)}
        for name, (lo, hi) in bounds.items():
            row[f'{name}_events'] = ids.reindex(hourly.index[lo:hi]).nunique()
        sensitivity_rows.append(row)
    sensitivity = pd.DataFrame(sensitivity_rows)

    gap_rows = []
    for missing in gaps:
        before, after = raw.index[raw.index.searchsorted(missing)-1], raw.index[raw.index.searchsorted(missing)]
        cutoff = missing.ceil('h')
        gap_rows.append(dict(missing_time_utc=missing, previous_raw_time_utc=before,
                             previous_stage_ft=raw.at[before, 'stage_ft'],
                             previous_approval_status=raw.at[before, 'approval_status'],
                             previous_qualifier=raw.at[before, 'qualifier'],
                             next_raw_time_utc=after, next_stage_ft=raw.at[after, 'stage_ft'],
                             next_approval_status=raw.at[after, 'approval_status'],
                             next_qualifier=raw.at[after, 'qualifier'], hourly_cutoff_utc=cutoff,
                             hourly_source_count=hourly.at[cutoff, 'n_source_in_hour'],
                             hourly_observed_time_utc=hourly.at[cutoff, 'observed_time_utc'],
                             hourly_stage_ft=hourly.at[cutoff, 'stage_ft']))
    gaps_table = pd.DataFrame(gap_rows)

    extreme_rows = []
    for scope, series in [('raw', raw.stage_ft), ('hourly', hourly.stage_ft)]:
        for kind, value in [('minimum', series.min()), ('maximum', series.max())]:
            for timestamp in series[series.eq(value)].index:
                source_time = timestamp if scope == 'raw' else hourly.at[timestamp, 'observed_time_utc']
                extreme_rows.append(dict(scope=scope, extreme=kind, time_utc=timestamp,
                                         source_time_utc=source_time, stage_ft=value,
                                         **raw.loc[source_time, ['unit', 'approval_status', 'qualifier', 'time_series_id']].to_dict()))
    extremes = pd.DataFrame(extreme_rows)
    boundaries = hourly.iloc[[0, -1]].reset_index()
    boundaries.insert(0, 'boundary', ['first', 'last'])
    boundaries['source_minutes_before_cutoff'] = [offsets.iloc[0] / pd.Timedelta(minutes=1),
                                                  offsets.iloc[-1] / pd.Timedelta(minutes=1)]

    assert len(raw) == len(expected) - len(gaps) + len(off_grid)
    assert annual.hours_ge_7_0_ft.sum() == len(flood) == events.flood_hours.sum() == split_summary.flood_hours.sum()
    assert annual.events_started.sum() == len(events) == split_summary.events_started.sum()
    assert split_by_event.flood_hours.sum() == len(flood)
    for name, table in [('annual_thresholds', annual), ('flood_events', events), ('chronological_split', split_summary),
                        ('event_split_membership', split_by_event), ('event_separation', sensitivity),
                        ('raw_gaps', gaps_table), ('raw_off_grid', raw.loc[off_grid].reset_index()),
                        ('stage_extremes', extremes), ('hourly_boundaries', boundaries)]:
        table.to_csv(metrics / f'qc_{name}.csv', index=False)

    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, ax = plt.subplots(figsize=(14, 5.5), layout='constrained')
    ax.plot(raw.index, raw.stage_ft, color='#126487', linewidth=.65, label='15-minute gage height')
    colors = ('#8b6b00', '#a64b16', '#aa2e46', '#662a74')
    categories = ('Action', 'Minor flood', 'Moderate flood', 'Major flood')
    for level, color, category in zip(thresholds, colors, categories):
        ax.axhline(level, color=color, lw=1.2, ls='--', label=f'{category} ({level:.1f} ft)')
    ax.set_xticks([pd.Timestamp(f'{year}-01-01', tz='UTC') for year in years])
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.set_xlim(start, stop)
    ax.set(title='Passaic River at Little Falls | USGS 01389500', ylabel='Gage height (ft)', xlabel='Observation time (UTC)')
    ax.grid(axis='y', alpha=.2)
    ax.legend(ncol=3, loc='lower left', framealpha=.96)
    fig.savefig(figures / 'qc_stage_2021_2025.png', dpi=180)
    plt.close(fig)

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(12, 4.6), layout='constrained')
    x = np.arange(len(annual))
    for i, (level, color) in enumerate(zip(thresholds, colors)):
        label = str(level).replace('.', '_')
        ax.bar(x + (i-1.5)*.19, annual[f'hours_ge_{label}_ft'], width=.18, color=color, label=f'{level:.1f} ft')
    ax.set_xticks(x, annual.year)
    ax.set(title='Hours at or above flood thresholds', ylabel='Hourly rows')
    ax.legend(ncol=2, frameon=False)
    bars = bx.bar(split_summary.split.str.capitalize(), split_summary.events_with_flood_hours, color='#126487', width=.55)
    bx.bar_label(bars, labels=[f'{c} events / {h} h' for c, h in zip(split_summary.events_with_flood_hours, split_summary.flood_hours)],
                 padding=4)
    bx.set_ylim(0, max(split_summary.events_with_flood_hours)*1.32)
    bx.set(title='Flood episodes represented in each split', ylabel='Independent episodes by the 24-hour rule')
    for axis in (ax, bx):
        axis.grid(axis='y', alpha=.2)
        axis.set_axisbelow(True)
    fig.savefig(figures / 'qc_thresholds_and_split_summary.png', dpi=180)
    plt.close(fig)

    for path, digest in original.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    for name, table in [('ANNUAL', annual), ('EVENTS', events), ('SPLITS', split_summary), ('MEMBERSHIP', split_by_event),
                        ('SENSITIVITY', sensitivity), ('GAPS', gaps_table), ('EXTREMES', extremes), ('BOUNDARIES', boundaries)]:
        print(name)
        print(table.to_string(index=False))
    print('raw expected', len(expected), 'off grid', len(off_grid), 'source flags', raw.approval_status.value_counts(dropna=False).to_dict(),
          'qualifiers', raw.qualifier.value_counts(dropna=False).to_dict())
    print('partial buckets', hourly.n_source_in_hour.value_counts().to_dict())
    print('source offsets', offsets.value_counts().to_dict())


if __name__ == '__main__':
    main()
