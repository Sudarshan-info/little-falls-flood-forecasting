"""Create an hourly river-stage series. Never invent missing readings by interpolation."""
import argparse
import json
import numpy as np
import pandas as pd
from . import config as C
from .utils import stage_class

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--synthetic', action='store_true', help='Use separate synthetic inputs and outputs.')
    args = parser.parse_args()
    prefix = 'synthetic_' if args.synthetic else ''
    src = C.RAW / f'{prefix}measurements.csv'
    if not src.exists():raise SystemExit('No measurements.csv. Run fetch, or download data first.')
    raw = pd.read_csv(src, dtype={'time_series_id': str})
    if not args.synthetic and 'approval_status' in raw and raw['approval_status'].eq('synthetic').any():
        raise SystemExit('Synthetic rows found in measurements.csv. Run demo_synthetic, then prepare --synthetic.')
    for required in ['time_utc','stage_ft']:
        if required not in raw:raise ValueError(f'Missing {required}; expected canonical USGS schema. Inspect your CSV.')
    if 'unit' in raw and not raw['unit'].dropna().astype(str).str.lower().isin(['ft','feet']).all():
        raise ValueError('Units are not all feet. Stop and convert using verified units.')
    raw['time_utc'] = pd.to_datetime(raw['time_utc'], format='ISO8601', utc=True, errors='coerce')
    raw['stage_ft'] = pd.to_numeric(raw['stage_ft'], errors='coerce')
    invalid = raw.time_utc.isna() | ~np.isfinite(raw.stage_ft)
    print('Invalid source rows:', invalid.sum())
    raw = raw.loc[~invalid].sort_values('time_utc', kind='stable')
    if raw.empty:
        raise SystemExit('No valid timestamps and finite stage readings in the input CSV.')
    if 'time_series_id' in raw and raw['time_series_id'].dropna().nunique()>1:
        ids=raw.groupby('time_series_id')['stage_ft'].count().sort_values(ascending=False)
        print('MULTIPLE TIME SERIES DETECTED:',ids.to_dict())
        print('Selecting most-populated time series for initial analysis; check if primary sensor.')
        raw=raw.loc[raw.time_series_id==ids.index[0]].copy()
    duplicates=raw.time_utc.duplicated().sum()
    print('Duplicate source timestamps:',duplicates)
    raw=raw.drop_duplicates('time_utc',keep='last').set_index('time_utc')
    # A row labeled 11:00 only uses readings in (10:00, 11:00].
    hourly = raw.assign(observed_time_utc=raw.index).resample('h', closed='right', label='right')
    out = hourly.agg(stage_ft=('stage_ft', 'last'),
                     n_source_in_hour=('stage_ft', 'count'),
                     observed_time_utc=('observed_time_utc', 'last'))
    h = out.stage_ft
    out['observed_class']=[stage_class(float(x)) for x in out.stage_ft]
    out.to_csv(C.PROCESSED / f'{prefix}hourly_stage.csv', index_label='time_utc')
    qc={'first_hour_utc':str(h.index.min()),'last_hour_utc':str(h.index.max()),
        'source_rows':int(len(raw)),'hourly_rows':int(len(out)),
        'hourly_missing_n':int(h.isna().sum()),'hourly_missing_pct':round(float(h.isna().mean()*100),3),
        'hours_with_ge_7ft':int((h>=C.MINOR_STAGE).sum()),'stage_min_ft':float(h.min()),'stage_max_ft':float(h.max())}
    with open(C.METRICS / f'{prefix}data_quality.json', 'w') as f:
        json.dump(qc,f,indent=2)
    print(json.dumps(qc,indent=2))
    if qc['hours_with_ge_7ft']==0:print('STOP: NO FLOOD HOURS: extend years or choose a new station.')
    if qc['hourly_missing_pct']>2:print('STOP: >2% missing; inspect before claiming results. Models will skip/handle missing.')
if __name__=='__main__':main()
