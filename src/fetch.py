"""Modern USGS OGC API download of continuous stage observations. Internet required.
Run a small date range first. This endpoint can change; compare against official download site.
"""
import argparse
from datetime import date
from pathlib import Path
from tempfile import NamedTemporaryFile
import time
from urllib.parse import urljoin
import pandas as pd
import requests
from . import config as C
BASE = 'https://api.waterdata.usgs.gov/ogcapi/v1/collections/continuous/items'
FIELDS = {
    'time_utc': 'time',
    'stage_ft': 'value',
    'unit': 'unit_of_measure',
    'approval_status': 'approval_status',
    'qualifier': 'qualifier',
    'time_series_id': 'time_series_id',
}


def utc_bounds(begin, end):
    return pd.Timestamp(begin, tz='UTC'), pd.Timestamp(end, tz='UTC') + pd.Timedelta(days=1)


def within_dates(rows, begin, end):
    first, stop = utc_bounds(begin, end)
    times = pd.to_datetime(rows['time_utc'], format='ISO8601', utc=True)
    rows = rows.assign(time_utc=times)
    return rows.loc[(times >= first) & (times < stop)]


def replace_csv(rows, path):
    # Publish a complete file after closing the temporary file on Windows.
    with NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        rows.to_csv(temporary, index=False)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)

def months(start,end):
    cur = pd.Timestamp(start).replace(day=1)
    stop = pd.Timestamp(end)
    while cur <= stop:
        nxt=cur+pd.offsets.MonthBegin(1)
        a=max(pd.Timestamp(start),cur)
        b=min(stop,nxt-pd.Timedelta(days=1))
        yield a.strftime('%Y-%m-%d'),b.strftime('%Y-%m-%d'),cur.strftime('%Y-%m')
        cur=nxt

def fetch_month(a,b):
    url=BASE
    first, stop = utc_bounds(a, b)
    params={'f':'json','monitoring_location_id':C.SITE,'parameter_code':C.PARAMETER,
            'datetime':f'{first:%Y-%m-%dT%H:%M:%SZ}/{stop:%Y-%m-%dT%H:%M:%SZ}','limit':10000}
    rows=[]; pages=0
    while url:
        for attempt in range(4):
            try:
                r=requests.get(url,params=params,timeout=90)
                r.raise_for_status(); payload=r.json();break
            except (requests.RequestException,ValueError):
                if attempt==3:raise
                time.sleep(2**attempt)
        for f in payload.get('features',[]):
            p=f.get('properties',{})
            if p.get('monitoring_location_id')!=C.SITE or str(p.get('parameter_code'))!=C.PARAMETER:
                continue
            missing = [key for key in ('time', 'unit_of_measure', 'time_series_id') if not p.get(key)]
            if 'value' not in p:
                missing.append('value')
            if missing:
                raise ValueError(f'Continuous response is missing required fields: {", ".join(missing)}')
            rows.append({column: p.get(field) for column, field in FIELDS.items()})
        pages+=1
        following=next((i.get('href') for i in payload.get('links',[]) if i.get('rel')=='next'),None)
        if not following:break
        if pages>=250:raise RuntimeError('Too many pages; check query filters before continuing.')
        url=urljoin(url, following); params=None
    return within_dates(pd.DataFrame(rows, columns=list(FIELDS)), a, b)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--start',required=True,help='YYYY-MM-DD')
    p.add_argument('--end',required=True,help='YYYY-MM-DD')
    p.add_argument('--overwrite', action='store_true', help='Replace an existing measurements.csv.')
    a=p.parse_args()
    try:
        start, end = date.fromisoformat(a.start), date.fromisoformat(a.end)
    except ValueError:
        p.error('--start and --end must be dates in YYYY-MM-DD format.')
    if (start.isoformat(), end.isoformat()) != (a.start, a.end):
        p.error('--start and --end must be dates in YYYY-MM-DD format.')
    today = pd.Timestamp.now(tz='UTC').date()
    if start > end or end > today:
        p.error('Require start <= end <= today (UTC).')
    target = C.RAW / 'measurements.csv'
    if target.exists() and not a.overwrite:
        p.error('measurements.csv already exists. Use --overwrite to replace it explicitly.')

    parts = []
    for begin, finish, month in months(a.start, a.end):
        path = C.RAW / f'usgs_{C.SITE}_{C.PARAMETER}_{begin}_{finish}.csv'
        cacheable = finish < today.isoformat()
        if cacheable and path.exists():
            rows = within_dates(pd.read_csv(path, dtype={'time_series_id': str}), begin, finish)
            print('Using cached', path.name)
        else:
            rows = fetch_month(begin, finish)
            print('Downloaded', month, 'rows', len(rows))
            if cacheable and not rows.empty:
                replace_csv(rows, path)
        if not rows.empty:
            parts.append(rows)
    if not parts:raise SystemExit('No data downloaded. Use the USGS site and check station/parameter.')
    all_rows = pd.concat(parts, ignore_index=True).drop_duplicates(
        subset=['time_utc', 'time_series_id', 'stage_ft'])
    all_rows = all_rows.sort_values(['time_utc', 'time_series_id'])
    if a.overwrite:
        replace_csv(all_rows, target)
    else:
        try:
            all_rows.to_csv(target, index=False, mode='x')
        except FileExistsError:
            p.error('measurements.csv already exists. Use --overwrite to replace it explicitly.')
    print('Saved data/raw/measurements.csv; rows:',len(all_rows),'time series:',all_rows.time_series_id.nunique())
    if all_rows['unit'].dropna().nunique()>1:print('STOP: mixed measurement units, inspect before preparing.')
if __name__=='__main__':main()
