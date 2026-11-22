"""Audit the isolated October 2007–2014 Little Falls exploration dataset."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config as C
from .checkpoint6 import HOUR, QUARTER, gap_runs, read_observations
from .prepare import hourly_stage
from .qc import flood_event_ids
from .utils import stage_class


START = pd.Timestamp('2007-10-01', tz='UTC')
STOP = pd.Timestamp('2015-01-01', tz='UTC')


def classify_events(stage):
    """Keep uncertain observed episodes in the catalog but out of confirmed counts."""
    above = stage[stage.ge(C.MINOR_STAGE)]
    candidates = list(above.groupby(flood_event_ids(above.index, 24)))
    confirmed_dry = stage.lt(C.MINOR_STAGE).rolling(24).sum().eq(24)
    starts = [readings.index[0] for _, readings in candidates]
    ends = [readings.index[-1] for _, readings in candidates]
    columns = ['candidate_id', 'status', 'reason', 'start_utc', 'end_utc', 'start_year',
               'duration_hours_inclusive', 'hours_ge_7_ft', 'bridged_below_threshold_hours',
               'unknown_hours_inside', 'peak_stage_ft', 'first_peak_utc']
    rows = []
    for i, (identifier, readings) in enumerate(candidates):
        first, last = starts[i], ends[i]
        prior = ends[i - 1] + HOUR if i else stage.index[0]
        following = starts[i + 1] - HOUR if i + 1 < len(starts) else stage.index[-1]
        interval = stage.loc[first:last]
        reasons = []
        if not confirmed_dry.loc[prior + 23 * HOUR:first - HOUR].any():
            reasons.append('no verified 24-hour separation before')
        if not confirmed_dry.loc[last + 24 * HOUR:following].any():
            reasons.append('no verified 24-hour separation after')
        neighbors = stage.reindex([first - HOUR, last + HOUR])
        if not neighbors.lt(C.MINOR_STAGE).all():
            reasons.append('unobserved onset or end boundary')
        if interval.isna().any():
            reasons.append('missing hours within episode')
        rows.append([identifier, 'uncertain' if reasons else 'confirmed', '; '.join(reasons),
                     first, last, first.year, len(interval), len(readings),
                     int(interval.lt(C.MINOR_STAGE).sum()), int(interval.isna().sum()),
                     readings.max(), readings.idxmax()])
    return pd.DataFrame(rows, columns=columns)


def make_plot(stage, confirmed, output):
    fig, ax = plt.subplots(figsize=(14, 5.3), layout='constrained')
    ax.plot(stage.index, stage, lw=.75, color='#155a79', label='Hourly gage height')
    for height, name, color in [(C.MINOR_STAGE, 'Current minor flood stage', '#a52d26'),
                                (C.MAJOR_STAGE, 'Current major flood stage', '#77508b')]:
        ax.axhline(height, ls='--', lw=1, color=color, label=f'{name} ({height:g} ft)')
    ax.scatter(confirmed.first_peak_utc, confirmed.peak_stage_ft, s=22, zorder=3,
               color='#242b32', label='Confirmed event peak')
    ax.set(title='Passaic River at Little Falls | Historical stage, Oct 2007–Dec 2014',
           xlim=(START, STOP), xlabel='Hourly cutoff (UTC)', ylabel='Gage height (ft)')
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.grid(axis='y', alpha=.22)
    ax.legend(loc='upper left', ncols=2)
    fig.savefig(output / 'stage_2007_2014.png', dpi=170)
    plt.close(fig)


def compare_training(study, confirmed):
    previous = C.ROOT / 'outputs' / 'checkpoint6'
    splits = pd.read_csv(previous / 'chronological_split.csv').set_index('split')
    training = splits.loc['train']
    modern = pd.read_csv(previous / 'flood_event_details.csv')
    modern = modern.loc[modern.start_year.between(2015, 2022)]
    verified = read_observations(C.PROCESSED / 'hourly_stage.csv')
    seam = pd.concat([study.stage_ft.tail(24), verified.stage_ft.head(24)])
    seam_continuous = (
        seam.index.equals(pd.date_range(STOP - 24 * HOUR, STOP + 23 * HOUR, freq='h'))
        and seam.lt(C.MINOR_STAGE).all()
    )
    return pd.DataFrame([
        {'period': '2007-10 through 2014', 'observed_events': len(confirmed),
         'observed_flood_hours': int(study.stage_ft.ge(C.MINOR_STAGE).sum()),
         'highest_event_peak_ft': confirmed.peak_stage_ft.max()},
        {'period': '2015 through 2022', 'observed_events': training.events_with_flood_hours,
         'observed_flood_hours': training.flood_hours, 'highest_event_peak_ft': modern.peak_ft.max()},
    ]), seam_continuous


def main():
    data = C.ROOT / 'data' / 'exploration' / '2007_2014'
    output = C.ROOT / 'outputs' / 'historical_extension'
    raw = read_observations(data / 'measurements.csv')
    if raw.index.hasnans or not raw.index.is_unique or not raw.index.is_monotonic_increasing:
        raise ValueError('Historical timestamps are invalid, duplicated, or out of order.')
    if not ((raw.index >= START) & (raw.index < STOP)).all() or not raw.unit.eq('ft').all():
        raise ValueError('Historical observations are outside the requested period or not in feet.')
    if raw.time_series_id.isna().any() or raw.time_series_id.nunique() != 1:
        raise ValueError('Multiple or missing series IDs require a sensor and datum review before aggregation.')
    raw['stage_ft'] = pd.to_numeric(raw.stage_ft, errors='coerce')
    valid = raw.loc[np.isfinite(raw.stage_ft)]
    index = pd.date_range(START, STOP, freq='h', name='time_utc')
    hourly = hourly_stage(valid).reindex(index)
    hourly['n_source_in_hour'] = hourly.n_source_in_hour.fillna(0).astype(int)
    hourly['observed_class'] = hourly.stage_ft.map(stage_class)
    offsets = hourly.index - hourly.observed_time_utc
    if not offsets.dropna().between(pd.Timedelta(0), HOUR, inclusive='left').all():
        raise ValueError('An hourly value uses an observation from outside its cutoff interval.')
    study = hourly.iloc[:-1]
    events = classify_events(study.stage_ft)
    confirmed = events.loc[events.status.eq('confirmed')]
    comparison, seam_dry = compare_training(study, confirmed)
    annual = []
    for year, rows in study.groupby(study.index.year):
        readings = valid.loc[valid.index.year == year]
        stage = rows.stage_ft
        gaps = gap_runs(rows.index[stage.isna()], HOUR)
        annual.append({
            'year': year, 'valid_raw_measurements': len(readings), 'expected_study_hours': len(rows),
            'valid_study_hours': int(stage.notna().sum()), 'missing_hours': int(stage.isna().sum()),
            'longest_missing_run_hours_within_year': int(gaps.missing_slots.max()) if len(gaps) else 0,
            'min_hourly_stage_ft': stage.min(), 'max_hourly_stage_ft': stage.max(),
            'hours_ge_7_ft': int(stage.ge(C.MINOR_STAGE).sum()),
            'confirmed_flood_events_started': int(confirmed.start_year.eq(year).sum()),
            'uncertain_candidates_started': int(events.loc[events.status.eq('uncertain'), 'start_year'].eq(year).sum()),
        })
    annual = pd.DataFrame(annual)
    months = pd.date_range(START, STOP, freq='MS', inclusive='left')
    monthly = raw.resample('MS').size().reindex(months, fill_value=0).rename('source_rows').rename_axis('month_utc')
    flags = raw.value_counts(['time_series_id', 'unit', 'approval_status', 'qualifier'], dropna=False)
    expected = pd.date_range(START, STOP, freq='15min', inclusive='left')
    missing_quarters = expected.difference(valid.index)
    off_grid = valid.index.difference(expected)
    gaps = gap_runs(study.index[study.stage_ft.isna()], HOUR)
    output.mkdir(parents=True, exist_ok=True)
    hourly.to_csv(data / 'hourly_stage.csv')
    for name, table in [('annual_summary', annual), ('flood_events', events), ('hourly_gaps', gaps),
                        ('monthly_coverage', monthly.reset_index()),
                        ('source_flags', flags.rename('rows').reset_index()),
                        ('training_comparison', comparison),
                        ('raw_quarter_hour_gaps', gap_runs(missing_quarters, QUARTER)),
                        ('off_grid_readings', valid.loc[off_grid].reset_index())]:
        table.to_csv(output / f'{name}.csv', index=False)
    make_plot(study.stage_ft, confirmed, output)
    print('Raw first and last:', raw.index[0], raw.index[-1])
    print('Raw rows, invalid stage readings:', len(raw), len(raw) - len(valid))
    print('Series and flags:', flags.to_dict())
    print('Cadence:', raw.index.to_series().diff().value_counts().head(10).to_dict())
    print('Empty months:', monthly.index[monthly.eq(0)].tolist())
    print('Expected quarter-hours, missing, off-grid:', len(expected), len(missing_quarters), len(off_grid))
    print('Study hours, missing:', len(study), study.stage_ft.isna().sum())
    print('Verified dry 24 hours on either side of the historical seam:', seam_dry)
    print(comparison.to_string(index=False))
    print('First and last raw extremes:', valid.stage_ft.idxmin(), valid.stage_ft.min(),
          valid.stage_ft.idxmax(), valid.stage_ft.max())
    print(annual.to_string(index=False))
    print(events.to_string(index=False))
    print(gaps.nlargest(8, 'missing_slots').to_string(index=False))


if __name__ == '__main__':
    main()
