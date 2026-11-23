"""Bounded expanding-window optimization confined to observations through 2023."""

import argparse
from importlib.metadata import version
import json
import math
from pathlib import Path
import platform
import time
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.statespace.kalman_filter import MEMORY_CONSERVE, MEMORY_NO_FORECAST_MEAN, MEMORY_NO_STD_FORECAST

from .experiment import C, HASHES, read_json, sha256, write_json
from .utils import complete_window_targets


RUN = C.ROOT / 'outputs' / 'experiments' / 'stage_v2_optimization'
YEARS = (2014, 2018, 2021, 2022, 2023)
END_2023_ROWS = 142464
ORDERS = [(p, 0, 0) for p in (1, 2, 3, 4, 6, 8, 12, 24)]
ORDERS += [(0,1,0), (1,1,0), (2,1,0), (4,1,0), (0,1,2), (1,1,1), (2,1,1), (3,1,2), (2,0,1)]


def load_stage():
    raw, hourly = C.RAW/'measurements.csv', C.PROCESSED/'hourly_stage.csv'
    if (sha256(raw), sha256(hourly)) != HASHES:
        raise ValueError('Canonical dataset hashes changed.')
    data = pd.read_csv(hourly, nrows=END_2023_ROWS, parse_dates=['time_utc','observed_time_utc']).set_index('time_utc')
    expected = pd.date_range('2007-10-01', '2023-12-31 23:00', freq='h', tz='UTC')
    if not data.index.equals(expected):
        raise ValueError('Historical grid is incomplete or extends beyond validation.')
    known = data.stage_ft.notna()
    measured = data.loc[known,'observed_time_utc']
    lag = measured.index-measured
    if not known.equals(data.observed_time_utc.notna()) or not lag.between(
            pd.Timedelta(0),pd.Timedelta(hours=1),inclusive='left').all():
        raise ValueError('Observation timing fails the causal cutoff rule.')
    return data.stage_ft


def make_folds(stage):
    for year in YEARS:
        lo = stage.index.searchsorted(pd.Timestamp(f'{year}-01-01',tz='UTC'))
        hi = stage.index.searchsorted(pd.Timestamp(f'{year+1}-01-01',tz='UTC'))
        if lo < 8760 or hi <= lo:
            raise ValueError(f'Insufficient history for {year}.')
        train = stage.iloc[:lo]
        yield year, lo, hi, train.diff().quantile(.99)


def trailing_block(stage, cap):
    last_missing = np.flatnonzero(stage.isna())
    start = last_missing[-1]+1 if len(last_missing) else 0
    return stage.iloc[max(start,len(stage)-cap):]


def score(stage, prediction, lo, hi, rise):
    truth = stage.iloc[lo:hi]
    common = complete_window_targets(stage.iloc[:hi],48)[lo:hi]
    prediction = prediction.reindex(truth.index).where(common)
    err = (prediction-truth).dropna()
    prior = stage.shift().loc[err.index]
    masks = {'overall':pd.Series(True,index=err.index), 'normal':truth.loc[err.index].lt(6.5),
             'rapid':truth.loc[err.index].sub(prior).ge(rise),
             'near7':truth.loc[err.index].sub(7).abs().le(.5),
             'flood':truth.loc[err.index].ge(7)}
    summary = {}
    for label, mask in masks.items():
        sample = err.loc[mask]
        summary.update({f'{label}_n':len(sample),f'{label}_sse':float(sample.pow(2).sum()),
                        f'{label}_sae':float(sample.abs().sum()),f'{label}_bias':float(sample.mean()),
                        f'{label}_rmse':float(np.sqrt(sample.pow(2).mean())),
                        f'{label}_mae':float(sample.abs().mean())})
    pos = truth[truth.ge(7)]
    groups = pos.index.to_series().diff().ge(pd.Timedelta(hours=25)).cumsum()
    details=[]
    for _, event in pos.groupby(groups):
        preds=prediction.reindex(event.index)
        hits=preds[preds.ge(7)]
        details.append({'onset':str(event.index[0]),'end':str(event.index[-1]),
                        'hours':len(event),'covered_hours':int(preds.notna().sum()),
                        'peak_observed':float(event.max()),'prediction_at_crest':float(preds.at[event.idxmax()]),
                        'hit_hours':len(hits),'detected':bool(len(hits)),
                        'first_hit_delay_h':(hits.index[0]-event.index[0])/pd.Timedelta(hours=1) if len(hits) else None})
    return summary,details


def cached(run, family, name, year, intent, fit):
    folder=run/'trials'/family
    folder.mkdir(parents=True,exist_ok=True)
    path=folder/f'{name}_{year}.json'
    if path.exists():
        found=read_json(path)
        if found['intent']!=intent:
            raise ValueError(f'Trial specification changed: {path}')
        return found
    started=time.perf_counter()
    try:
        result=fit(path)
    except Exception as error:
        result={'status':'failed','error':f'{type(error).__name__}: {error}'}
    row={'candidate':name,'year':year,'intent':intent,**result,'seconds':time.perf_counter()-started}
    write_json(path,row)
    print(f"{family} {name} {year}: {row['status']} {row['seconds']:.2f}s",flush=True)
    return row


def classical_predict(stage, fold, order, cap, path):
    _,lo,hi,rise=fold
    train=trailing_block(stage.iloc[:lo],cap).asfreq('h')
    if len(train)<1000:
        raise ValueError(f'Only {len(train)} contiguous fitting hours.')
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        trend='c' if order[1]==0 else 'n'
        fit=SARIMAX(train,order=order,trend=trend).fit(disp=False,maxiter=24)
        pred=fixed_predictions(stage,lo,hi,order,dict(fit.params))
    metrics,events=score(stage,pred,lo,hi,rise)
    return {'status':'ok' if fit.mle_retvals.get('converged') else 'nonconverged','common48':True,
            'metrics':metrics,'events':events,'warnings':[str(w.message) for w in caught],
            'fit_start':str(train.index[0]),'fit_hours':len(train),'fit_flood_hours':int(train.ge(7).sum()),
            'parameters':dict(fit.params)}


def fixed_predictions(stage,lo,hi,order,parameters):
    history=stage.iloc[:hi].asfreq('h')
    model=SARIMAX(history,order=order,trend='c' if order[1]==0 else 'n')
    model.update([parameters[p] for p in model.param_names])
    # Retain the Kalman one-step forecast means and omit large state arrays.
    flags=(MEMORY_CONSERVE & ~MEMORY_NO_FORECAST_MEAN) | MEMORY_NO_STD_FORECAST
    predicted=model.ssm.filter(conserve_memory=flags).forecasts[0,lo:hi]
    return pd.Series(predicted,index=history.index[lo:hi])


def aggregate(rows, persistence=None):
    result=[]
    for name in sorted({r['candidate'] for r in rows}):
        group=[r for r in rows if r['candidate']==name]
        good=[r for r in group if r['status']=='ok']
        item={'candidate':name,'ok_folds':len(good),'attempted_folds':len(group),'overall_rmse':np.nan,
              'seconds':sum(r['seconds'] for r in group)}
        if good:
            for metric in ('overall','normal','rapid','near7','flood'):
                n=sum(r['metrics'][f'{metric}_n'] for r in good)
                item[f'{metric}_n']=n
                item[f'{metric}_rmse']=math.sqrt(sum(r['metrics'][f'{metric}_sse'] for r in good)/n) if n else np.nan
                item[f'{metric}_mae']=sum(r['metrics'][f'{metric}_sae'] for r in good)/n if n else np.nan
            rmse=[r['metrics']['overall_rmse'] for r in good]
            item.update(worst_year_rmse=max(rmse),year_rmse_sd=float(np.std(rmse)),
                        events=sum(len(r['events']) for r in good),
                        events_detected=sum(e['detected'] for r in good for e in r['events']))
            item['eligible']=(len(good)==len(YEARS) and (persistence is None or
                               item['normal_rmse']<=1.05*persistence))
        else:
            item['eligible']=False
        result.append(item)
    return pd.DataFrame(result).sort_values(['eligible','overall_rmse'],ascending=[False,True])


def save_tables(run,name,rows,reference=None):
    table=aggregate(rows,reference)
    table.to_csv(run/f'{name}_summary.csv',index=False)
    pd.DataFrame([{k:v for k,v in row.items() if k not in ('events','metrics')}|row.get('metrics',{})
                  for row in rows]).to_csv(run/f'{name}_folds.csv',index=False)
    pd.DataFrame([{'candidate':row['candidate'],'year':row['year'],**e} for row in rows
                  for e in row.get('events',[])]).to_csv(run/f'{name}_events.csv',index=False)
    return table


def run_classical(stage,folds,run):
    base=[]
    for fold in folds:
        year,lo,hi,rise=fold
        for name,lag in [('persistence',1),('daily_naive',24)]:
            def fit(_):
                past=stage.iloc[:hi]
                pred=past.shift(lag).where(complete_window_targets(past,lag))
                metrics,events=score(stage,pred,lo,hi,rise)
                return {'status':'ok','common48':True,'metrics':metrics,'events':events}
            row=cached(run,'baseline',name,year,{'lag':lag},fit)
            base.append(row if row.get('common48') else row|fit(None))
    summary=save_tables(run,'baseline',base)
    reference=float(summary.set_index('candidate').at['persistence','normal_rmse'])
    rows=[]
    caps=(8000,32000)
    for order in ORDERS:
        for cap in caps if order in {(4,0,0),(8,0,0),(1,1,1),(2,1,1),(2,1,0)} else caps[:1]:
            name=f'{order[0]}_{order[1]}_{order[2]}_cap{cap}'
            for fold in folds:
                pilot=run/'trials'/'classical'/f'{name}_2014.json'
                prune=(order==(24,0,0) and fold[0]!=2014 and pilot.exists()
                       and read_json(pilot)['seconds']>25)
                row=cached(run,'classical',name,fold[0],{'order':list(order),'cap':cap},
                           lambda path:({'status':'pruned','reason':'AR(24) pilot exceeded 25 seconds'}
                                        if prune else classical_predict(stage,fold,order,cap,path)))
                if not row.get('common48') and 'parameters' in row and order!=(24,0,0):
                    _,lo,hi,rise=fold
                    pred=fixed_predictions(stage,lo,hi,order,row['parameters'])
                    metrics,events=score(stage,pred,lo,hi,rise)
                    row=row|{'metrics':metrics,'events':events,'common48':True}
                rows.append(row)
    save_tables(run,'classical',rows,reference)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=('classical','neural','all','report'),default='all')
    args=parser.parse_args()
    RUN.mkdir(parents=True,exist_ok=True)
    policy={'years':YEARS,'latest_observation_utc':'2023-12-31 23:00',
            'primary':'Pooled hourly validation RMSE across all five folds',
            'guardrail':'Normal-stage RMSE (target <6.5 ft) must be <=105% of persistence.',
            'secondary':'MAE, rapid rise (each fold training-only 99th percentile), near threshold (+/-0.5 ft), event overlap and per-year variability',
            'dataset_sha256':HASHES}
    path=RUN/'protocol.json'
    if not path.exists():
        write_json(path,policy)
    elif read_json(path)!=json.loads(json.dumps(policy)):
        raise ValueError('The saved evaluation protocol differs.')
    stage=load_stage()
    folds=list(make_folds(stage))
    write_json(RUN/'reproducibility.json',{
        'runtime':{'python':platform.python_version(),'platform':platform.system(),
                   'packages':{name:version(name) for name in ('numpy','pandas','statsmodels','torch','matplotlib')},
                   'torch_device':'cpu','torch_threads':2},
        'script_sha256':{name:sha256(C.ROOT/'src'/name)
                         for name in ('experiment_v2.py','experiment_v2_neural.py','utils.py')},
        'original_v1_git_commit':'dfa384eb200903d2eb90c9ebe4d8a418e504a87e',
        'command':r'.\.venv\Scripts\python.exe -B -m src.experiment_v2 --phase all',
        'resume':'Keep this directory and rerun. Completed trial JSON manifests are reused.',
    })
    write_json(RUN/'alignment.json', {'common_target_mask':'target and previous 48 hourly observations must be finite',
                                     'scope':'all point models and folds; interval variants share their own identical samples'})
    pd.DataFrame([{'year':year,'train_end':stage.index[lo-1],'validation_start':stage.index[lo],
                   'validation_end':stage.index[hi-1],'training_valid':stage.iloc[:lo].count(),
                   'validation_valid':stage.iloc[lo:hi].count(),'rapid_threshold_ft_h':rise}
                  for year,lo,hi,rise in folds]).to_csv(RUN/'folds.csv',index=False)
    if args.phase in ('classical','all'):
        run_classical(stage,folds,RUN)
    if args.phase in ('neural','all'):
        from .experiment_v2_neural import run_neural
        run_neural(stage,folds,RUN)
    if args.phase in ('report','all'):
        from .experiment_v2_neural import make_report
        make_report(RUN)


if __name__=='__main__':
    main()
