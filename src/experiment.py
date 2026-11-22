"""Run immutable, validation-selected one-hour stage experiments."""

import argparse
import json
import platform
from pathlib import Path
from time import perf_counter
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import acf, adfuller, kpss

from . import config as C
from .finalize_dataset import sha256
from .utils import complete_window_targets, forecast_splits, stats


HASHES = ('32bba60e465f52d601dc7a4b49d5c3d58d94cb548e1175cab26d97176080c11a',
          '41663c8fbe2b8dd29327c7318573a9d66f4d9236e2661bb5ace73e67efdc1a31')
BASELINES = {'persistence': 1, 'seasonal_24h': 24}
CLASSICAL = {'ma': [(0, 0, 1), (0, 0, 2)], 'ar': [(1, 0, 0), (4, 0, 0), (8, 0, 0)],
             'arima': [(0, 1, 0), (1, 1, 0), (2, 1, 0), (1, 1, 1), (2, 1, 1)]}


def write_json(path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2, default=str) + '\n', encoding='utf8')
    temporary.replace(path)


def read_json(path):
    return json.loads(path.read_text(encoding='utf8'))


def load_data(run):
    files = (C.RAW/'measurements.csv', C.PROCESSED/'hourly_stage.csv')
    if tuple(map(sha256, files)) != HASHES:
        raise ValueError('Canonical release hashes did not match.')
    data = pd.read_csv(files[1], parse_dates=['time_utc', 'observed_time_utc']).set_index('time_utc')
    bounds = forecast_splits(data.index)
    known = data.stage_ft.notna()
    source = data.loc[known, 'observed_time_utc']
    delta = source.index - source
    if not delta.between(pd.Timedelta(0), pd.Timedelta(hours=1), inclusive='left').all():
        raise ValueError('A source observation is outside its causal hourly bucket.')
    if not known.equals(data.observed_time_utc.notna()):
        raise ValueError('Stage and source timestamp validity disagree.')
    if not (data.stage_ft.count(), len(data), int(data.stage_ft.isna().sum())) == (159696, 160009, 313):
        raise ValueError('Canonical row and missingness totals changed.')
    preflight = {'dataset': 'dataset-v1.0.0', 'hashes_raw_hourly': HASHES,
                 'bounds_exclusive': bounds, 'hours_missing': 313,
                 'positioning': 'Hourly cutoff t uses source readings in (t-1h,t]. Target is the next cutoff.',
                 'python': platform.python_version(), 'statsmodels': statsmodels.__version__,
                 'numpy': np.__version__, 'pandas': pd.__version__}
    write_json(run/'preflight.json', preflight)
    return data.stage_ft, bounds


def latest_block(series, maximum):
    blocks = series.dropna().groupby(series.isna().cumsum())
    return list(blocks)[-1][1].tail(maximum)


def diagnostics(train, run):
    path = run/'stationarity.json'
    if path.exists():
        return read_json(path)
    clock = perf_counter()
    block = latest_block(train, 12000)
    diagnostics_rows = {}
    difference = block.diff().dropna()
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), layout='constrained')
    for row, (name, series) in enumerate([('level', block), ('difference', difference)]):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            a = adfuller(series, maxlag=24, autolag=None)
            k = kpss(series, nlags='auto')
        correlations = acf(series, nlags=72, fft=True)
        diagnostics_rows[name] = {'adf_stat': a[0], 'adf_p': a[1], 'adf_lag': a[2],
                                  'kpss_stat': k[0], 'kpss_p': k[1], 'kpss_lag': k[2],
                                  'acf_lag_1': correlations[1], 'acf_lag_24': correlations[24],
                                  'acf_lag_48': correlations[48], 'acf_lag_72': correlations[72],
                                  'warnings': [str(w.message) for w in caught]}
        plot_acf(series, lags=72, fft=True, ax=axes[row, 0], title=f'ACF | {name}')
        plot_pacf(series, lags=48, method='ywm', ax=axes[row, 1], title=f'PACF | {name}')
    fig.savefig(run/'training_acf_pacf.png', dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(2, 1, figsize=(13, 6), layout='constrained')
    axes[0].plot(train, lw=.65)
    axes[0].axhline(7, color='#a34336', ls='--')
    axes[0].set(title='Training stage', ylabel='Feet')
    axes[1].plot(train.diff(), lw=.6)
    axes[1].set(title='Adjacent hourly changes (missing values stay blank)', ylabel='Feet per hour')
    fig.savefig(run/'training_stage_and_changes.png', dpi=150)
    plt.close(fig)
    changed = diagnostics_rows['difference']
    daily_supported = all(changed[key] >= .15 for key in ('acf_lag_24', 'acf_lag_48'))
    result = {'block_n': len(block), 'block_start': str(block.index[0]), 'block_end': str(block.index[-1]),
              'statistics': diagnostics_rows, 'daily_supported': daily_supported,
              'daily_screen': 'Both differenced ACF(24) and ACF(48) must reach +0.15.',
              'seconds': perf_counter()-clock}
    if daily_supported:
        from statsmodels.tsa.seasonal import STL
        decomposition = STL(block, period=24, robust=True).fit()
        fig = decomposition.plot()
        fig.savefig(run/'training_daily_stl.png', dpi=150)
        plt.close(fig)
    write_json(path, result)
    return result


def slice_predictions(stage, bounds, key, lag):
    lo, hi = bounds[key]
    history = stage.iloc[:hi]
    return history.shift(lag).where(complete_window_targets(history, lag)).iloc[lo:hi]


def save_predictions(run, stage, predicted, name, split):
    actual = stage.loc[predicted.index]
    result = pd.DataFrame({'issued_utc': predicted.index - pd.Timedelta(hours=1),
                           'target_utc': predicted.index, 'actual_ft': actual.to_numpy(),
                           'predicted_ft': predicted.to_numpy()})
    result.to_csv(run/f'{split}_{name}_predictions.csv', index=False)
    return stats(actual, predicted)


def hydrograph(stage, forecasts, path, limit=450):
    fig, ax = plt.subplots(figsize=(13, 4), layout='constrained')
    ax.plot(stage.iloc[:limit], label='Observed', color='#20262e', lw=1.6)
    for name, values in forecasts.items():
        ax.plot(values.iloc[:limit], label=name, lw=1)
    ax.axhline(7, lw=.9, ls='--', color='#9b342d', label='Flood stage')
    ax.set(ylabel='Stage (ft)', xlabel='Target hour (UTC)')
    ax.legend(ncols=4)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def baseline_validation(stage, bounds, run):
    metrics = {}
    forecasts = {}
    for name, lag in BASELINES.items():
        clock = perf_counter()
        pred = slice_predictions(stage, bounds, 'val', lag)
        forecasts[name] = pred
        metrics[name] = {**save_predictions(run, stage, pred, name, 'validation'),
                         'seconds': perf_counter()-clock}
    hydrograph(stage.loc[pred.index], forecasts, run/'validation_baselines.png')
    write_json(run/'baseline_validation.json', metrics)


def classical_predict(fit, stage, bounds, split, lag):
    lo, hi = bounds[split]
    history = stage.iloc[:hi].asfreq('h')
    applied = fit.apply(history, refit=False)
    prediction = applied.get_prediction(start=lo, end=hi-1, information_set='predicted').predicted_mean
    mask = complete_window_targets(history, lag)[lo:hi]
    return prediction.where(mask)


def classical_search(stage, bounds, run, diag):
    start = perf_counter()
    block = latest_block(stage.iloc[:bounds['train'][1]], 8000).asfreq('h')
    trials = run/'classical_trials'
    trials.mkdir(exist_ok=True)
    candidates = [(family, order, (0,0,0,0)) for family, orders in CLASSICAL.items() for order in orders]
    if diag['daily_supported']:
        candidates.append(('sarima', (1,1,0), (1,0,0,24)))
    rows = []
    for family, order, seasonal in candidates:
        name = family+'_'+''.join(map(str,order))+'_'+''.join(map(str,seasonal))
        meta = trials/f'{name}.json'
        if meta.exists():
            rows.append(read_json(meta))
            continue
        clock = perf_counter()
        row = {'name': name,'family': family,'order': order,'seasonal':seasonal,
               'fit_n':len(block),'first_fit_hour':str(block.index[0]),'last_fit_hour':str(block.index[-1])}
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                trend = 'c' if order[1] == 0 else 'n'
                fit = SARIMAX(block, order=order, seasonal_order=seasonal, trend=trend).fit(disp=False,maxiter=24)
                lag = max(1, max(order)+max(seasonal[:3])*seasonal[3])
                pred = classical_predict(fit, stage, bounds, 'val', lag)
            residuals = fit.resid.iloc[max(24,lag):].dropna()
            res_acf = acf(residuals, nlags=24, fft=True)
            row.update(converged=bool(fit.mle_retvals.get('converged')), lag=lag,
                       warnings=[str(w.message) for w in caught], aic=fit.aic,
                       residual_acf_1=res_acf[1], residual_acf_24=res_acf[24],
                       residual_std_ft=residuals.std(), **stats(stage.loc[pred.index],pred))
            if row['converged'] and np.isfinite(row['RMSE_ft']):
                fit.save(trials/f'{name}.pkl')
        except Exception as error:
            row.update(converged=False,error=f'{type(error).__name__}: {error}')
        row['seconds'] = perf_counter()-clock
        write_json(meta,row)
        rows.append(row)
        print(name, row.get('RMSE_ft'), row.get('converged'), round(row['seconds'],2), flush=True)
    pd.DataFrame(rows).to_csv(run/'classical_search.csv', index=False)
    selected = {}
    from statsmodels.tsa.statespace.sarimax import SARIMAXResults
    for family in sorted({row['family'] for row in rows}):
        eligible = [row for row in rows if row['family']==family and row['converged']]
        if eligible:
            winner = min(eligible,key=lambda row:row['RMSE_ft'])
            selected[family] = winner
            fit = SARIMAXResults.load(trials/f"{winner['name']}.pkl")
            pred = classical_predict(fit,stage,bounds,'val',winner['lag'])
            save_predictions(run,stage,pred,family,'validation')
            hydrograph(stage.loc[pred.index],{family:pred},run/f'validation_{family}.png')
    write_json(run/'classical_selected.json',selected)
    write_json(run/'classical_timing.json',{'this_execution_seconds':perf_counter()-start,
                                           'total_candidate_seconds':sum(row['seconds'] for row in rows)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, help='Run ID inside outputs/experiments.')
    parser.add_argument('--phase', choices=['classical','neural','final','all'], default='all')
    args = parser.parse_args()
    if Path(args.run).name != args.run or args.run in ('.','..'):
        parser.error('Use a simple run ID.')
    run = C.ROOT/'outputs'/'experiments'/args.run
    run.mkdir(parents=True,exist_ok=True)
    stage,bounds = load_data(run)
    if args.phase in ('classical','all') and not (run/'freeze.json').exists():
        diag = diagnostics(stage.iloc[:bounds['train'][1]],run)
        if not (run/'baseline_validation.json').exists():
            baseline_validation(stage,bounds,run)
        if not (run/'classical_selected.json').exists():
            classical_search(stage,bounds,run,diag)
    if args.phase in ('neural','all'):
        from .experiment_neural import run_neural
        run_neural(stage,bounds,run)
    if args.phase in ('final','all'):
        from .experiment_evaluate import evaluate
        evaluate(stage,bounds,run)


if __name__ == '__main__':
    main()
