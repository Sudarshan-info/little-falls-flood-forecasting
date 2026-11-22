"""Freeze validation winners and score the held-out period once."""

from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from statsmodels.tsa.statespace.sarimax import SARIMAXResults

from .experiment import (BASELINES, HASHES, classical_predict, hydrograph, read_json,
                         save_predictions, sha256, slice_predictions, stats, write_json)
from .experiment_neural import predict_checkpoint, save_interval
from .qc import flood_event_ids
from .utils import binary_scores


HOUR = pd.Timedelta(hours=1)


def episodes(mask):
    times = mask.index[mask]
    return [list(group) for _, group in times.to_series().groupby(flood_event_ids(times, 24))]


def warning_metrics(actual, predicted, name):
    observed = actual.ge(7)
    alerts = predicted.ge(7)
    scores = binary_scores(observed,alerts)
    alert_episodes = episodes(alerts)
    truth = episodes(observed)
    alert_sets = [set(episode) for episode in alert_episodes]
    false = [episode for episode in alert_episodes if not observed.reindex(episode).any()]
    rows = []
    for number, event in enumerate(truth,1):
        matching = [episode for episode, candidates in zip(alert_episodes,alert_sets)
                    if candidates.intersection(event)]
        onset = event[0]
        issue = min((interval[0]-HOUR for interval in matching),default=pd.NaT)
        hit = min((time for interval in matching for time in interval if time in event),default=pd.NaT)
        rows.append({'model':name,'event':number,'onset_utc':onset,'end_utc':event[-1],
                     'peak_ft':actual.loc[event].max(),'detected':bool(matching),
                     'first_matched_alert_issue_utc':issue,'first_overlapping_alert_target_utc':hit,
                     'lead_hours_positive_is_early':(onset-issue)/HOUR,
                     'first_hit_target_delay_hours':(hit-onset)/HOUR})
    scores.update(model=name,events=len(truth),detected=sum(row['detected'] for row in rows),
                  missed=sum(not row['detected'] for row in rows),false_alarm_episodes=len(false),
                  forecast_alert_episodes=len(alert_episodes))
    return scores, rows, [{'model':name,'start_target_utc':e[0],'end_target_utc':e[-1],
                            'alert_hours':len(e),'first_issue_utc':e[0]-HOUR} for e in false]


def freeze_protocol(run, stage, bounds):
    path=run/'freeze.json'
    if path.exists():
        state=read_json(path)
        for checkpoint,digest in state['checkpoint_sha256'].items():
            if sha256(run/checkpoint)!=digest:
                raise ValueError(f'Frozen checkpoint no longer matches: {checkpoint}')
        return state
    classical=read_json(run/'classical_selected.json')
    neural=read_json(run/'neural_selected.json')
    quantile=read_json(run/'quantile_selected.json')
    if set(neural)!={'rnn','lstm'} or set(quantile)!={'lower','upper'}:
        raise RuntimeError('Validation selection has not finished.')
    # Interval validation is required before unlocking test, including on resume.
    interval_validation=read_json(run/'validation_quantile_metrics.json')
    checkpoints=[f"classical_trials/{row['name']}.pkl" for row in classical.values()]
    checkpoints.extend(f"neural_trials/{row['trial']}.pt" for row in (*neural.values(),*quantile.values()))
    cutoff=float(stage.iloc[:bounds['train'][1]].diff().quantile(.99))
    state={'dataset_sha256':HASHES,'classical':classical,'neural':neural,'quantile':quantile,
           'baseline_lags_hours':BASELINES,'scaler':read_json(run/'training_scaler.json'),
           'interval_validation':interval_validation,'checkpoint_sha256':{p:sha256(run/p) for p in checkpoints},
           'rapid_rise_cutoff_ft_per_hour_training_99th_percentile':cutoff,
           'evaluation':'Rolling one hour ahead, no test refitting; point comparisons share valid timestamps.',
           'warning':'Point forecasts >=7 ft; event matching requires overlap with an observed positive target.',
           'quantile_policy':'No recalibration. Order bounds for coverage and report raw crossings separately.',
           'mape_policy':'Exclude actual magnitude <= max(1e-8, 1% of the common sample mean absolute stage).'}
    write_json(path,state)
    return state


def validation_table(run, names):
    frames = {}
    for name in names:
        data = pd.read_csv(run/f'validation_{name}_predictions.csv',parse_dates=['target_utc']).set_index('target_utc')
        frames[name] = data.predicted_ft
    aligned=pd.concat(frames,axis=1).dropna()
    actual=data.actual_ft.loc[aligned.index]
    return pd.DataFrame([{'model':name,**stats(actual,aligned[name])} for name in names]).sort_values('RMSE_ft')


def flood_plot(actual, forecasts, comparisons, run):
    best=comparisons.iloc[0].model
    fig,axes=plt.subplots(3,1,figsize=(12,9),layout='constrained')
    for ax,event in zip(axes,episodes(actual.ge(7))):
        subset=actual.loc[event[0]-24*HOUR:event[-1]+24*HOUR]
        ax.plot(subset,label='Observed',color='#20313e',lw=1.6)
        for name in dict.fromkeys(['persistence',best]):
            ax.plot(forecasts[name].loc[subset.index],label=name,lw=1)
        ax.axhline(7,color='#a34236',ls='--')
        ax.axvline(event[0],color='#607788',ls=':')
        ax.set(ylabel='Feet',title=f'Observed onset {event[0]:%Y-%m-%d %H:%M UTC}')
    axes[0].legend(ncols=3)
    fig.savefig(run/'test_flood_hydrographs.png',dpi=150)
    plt.close(fig)


def markdown_table(frame, columns):
    def formatted(value):
        return f'{value:.5f}' if isinstance(value,float) else str(value)
    lines=['| '+' | '.join(columns)+' |','| '+' | '.join(['---']*len(columns))+' |']
    lines.extend('| '+' | '.join(formatted(row[col]) for col in columns)+' |' for _,row in frame.iterrows())
    return '\n'.join(lines)


def warning_timing_figure(events, path):
    table = events.pivot(index='model', columns='event', values='lead_hours_positive_is_early')
    fig, ax = plt.subplots(figsize=(9, 4.3), layout='constrained')
    table.plot.barh(ax=ax)
    ax.axvline(0, color='#394957', lw=1)
    ax.set(xlabel='Hours before observed onset (negative means late)', ylabel='Model',
           title='First matched warning issuance by observed flood event')
    ax.grid(axis='x', alpha=.2)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def report(run, freeze, val, test, warnings_table, event_rows, interval, timing):
    diag=read_json(run/'stationarity.json')
    level=diag['statistics']['level']
    diff=diag['statistics']['difference']
    fit=list(freeze['classical'].values())
    neural=list(freeze['neural'].values())
    configs=[f"- {r['family'].upper()} {tuple(r['order'])}; fit {r['fit_n']:,} contiguous training hours."
             for r in fit]
    configs.extend(f"- {r['cell'].upper()} {r['config']}; best epoch {r['best_epoch']}; seed {r['seed']}."
                   for r in neural)
    metrics=['model','n','RMSE_ft','MAE_ft','R2','MAPE_percent_excluding_near_zero']
    baseline=test.loc[test.model.eq('persistence'),'RMSE_ft'].iloc[0]
    improved=test.loc[test.RMSE_ft.lt(baseline),'model'].tolist()
    seed_check=pd.read_csv(run/'lstm_seed_sensitivity.csv')
    residuals=pd.DataFrame(fit)
    lead=event_rows.pivot(index='model',columns='event',values='lead_hours_positive_is_early')
    lead=lead.rename(columns=lambda number:f'event_{number}_lead_hours').reset_index().astype(object).fillna('missed')
    rises=pd.read_csv(run/'test_rapid_rises.csv')
    warning_timing_figure(event_rows,run/'test_warning_timing.png')
    contents=f"""# Little Falls one-hour forecasting research report

## Data and forecast protocol

The canonical release contains 160,008 study cutoffs and 313 missing hours. Training ends in 2022, validation is 2023, and test is 2024–2025. At cutoff t the last valid reading in (t−1 hour, t] is available in the archive. The target is H(t+1). Observations are released sequentially during rolling evaluation. No model parameters are learned from test. Missing targets or lookback hours disqualify explicit windows. The 184-hour and 42-hour outages remain unfilled. State-space filtering propagates through missing observations without substituting fabricated values. Publication latency is unknown. The model effects of rainfall, upstream releases and river regulation are not identifiable from stage alone.

## Training diagnostics

The last contiguous block contributes {diag['block_n']:,} hours from {diag['block_start']} through {diag['block_end']}. ADF uses 24 fixed lags and tests a unit-root null; KPSS tests a level-stationarity null. On stage levels, ADF p = {level['adf_p']:.6g}, KPSS p = {level['kpss_p']:.6g}, and lag-1 ACF = {level['acf_lag_1']:.4f}. On differences, ADF p = {diff['adf_p']:.6g}, KPSS p = {diff['kpss_p']:.6g}, and lag-24/48 ACFs are {diff['acf_lag_24']:.4f} / {diff['acf_lag_48']:.4f}. Tabulated KPSS boundary p-values are limits; see saved warnings. The persistent level series and the change-series diagnostics motivate testing d=1 as a practical specification; held-out validation determines useful forecasts. These tests do not establish hydrologic stationarity or justify extrapolating flood risk. Short-memory structure is also present, so AR and MA on levels were evaluated separately. The repeated daily-lag screen {'passed' if diag['daily_supported'] else 'did not pass'}; {'STL and a seasonal candidate were run' if diag['daily_supported'] else 'daily decomposition and SARIMA were omitted'}.

## Model development and validation

Persistence uses H(t); daily naive uses H(t−23h) for target H(t+1). The classical model equation is Y(u) = c + ΣφᵢY(u−i) + ε(u) + Σθⱼε(u−j), where Y=H for MA and AR and Y=ΔH for ARIMA with d=1. MA(q) models lagged innovations. Classical likelihood fits used the last 8,000 contiguous training hours to cap CPU time; post-fit filtering used past observations without refitting. The search retains failed or nonconverged candidates. In particular, AR(1) did not converge and was ineligible. The selected MA fit leaves substantial residual correlation at both inspected lags.

{markdown_table(residuals,['family','residual_acf_1','residual_acf_24','residual_std_ft'])}

Recurrent networks predict a scaled change from the latest stage. The forecast in feet is H(t) plus the predicted change times the training scale. Each tuned configuration used up to 18,000 windows sampled evenly across eligible training-window positions. Validation used all eligible windows. The optimizer is Adam, with gradient clipping at 1. Early stopping used patience 7. Smoke tests used 2,048 windows, two configurations and five epochs per cell. The recurrent search screened four configurations per cell at up to 18 epochs each. Seed 42 was fixed for selecting candidates and retained for the final checkpoints. Two independent LSTM repetitions measured validation seed sensitivity without changing the chosen model.

{markdown_table(seed_check,['seed','status','RMSE_ft','MAE_ft'])}

{chr(10).join(configs)}

{markdown_table(val,metrics)}

The freeze manifest records the winning configurations and checkpoint hashes before test scoring. MAPE filters near-zero values using the rule in that manifest.

## Final test, common targets

{markdown_table(test,metrics)}

Models with lower common-target test RMSE than persistence: {', '.join(improved) or 'none'}.

## Flood-warning performance

The binary warning rule is forecast H(t+1) ≥ 7 ft. The action level of 6.5 ft is descriptive. Observed and predicted episodes group positive targets unless separated by at least 24 below-threshold hours. A detection requires a predicted positive target to overlap an observed positive target. A false-alarm episode has no such overlap. Lead measures observed onset minus the issue time of the first alert in the matched forecast episode. The event table also records delay to the first overlapping target. These results are exploratory with only three independent test floods.

{markdown_table(warnings_table,['model','TP','FP','TN','FN','precision','recall','F1','detected','missed','false_alarm_episodes'])}

{markdown_table(lead,list(lead.columns))}

Rapid-rise comparisons use the 99th percentile of training hourly changes, {freeze['rapid_rise_cutoff_ft_per_hour_training_99th_percentile']:.4f} ft/h. The rise table records bias, MAE, and how often the forecast underestimates the observed hourly rise.

{markdown_table(rises,['model','rapid_rise_hours','underestimated_fraction','mean_error_ft','MAE_ft'])}

## Quantile intervals

Separate LSTMs use pinball loss max(qe,(q−1)e) where e = observed minus forecast, at q = 0.025 and 0.975. Validation selects stopping epochs. Bounds are ordered for scoring and raw crossing frequency is retained. No calibration was fitted. Coverage is empirical and a nominal 95% interval has no guarantee of delivering 95% coverage.

Validation PICP {freeze['interval_validation']['PICP']:.3%}; width {freeze['interval_validation']['MPIW_ft']:.4f} ft. Test PICP {interval['PICP']:.3%}; width {interval['MPIW_ft']:.4f} ft; crossing frequency {interval['crossing_pct']:.3f}%. For the {interval['rising_flood_n']} test hours with observed stage at or above flood level and rising, coverage was {interval['rising_flood_PICP']:.3%}, with mean width {interval['rising_flood_MPIW_ft']:.4f} ft.

## Runtime and reproducibility

Classical candidate time {timing.get('classical_trial_seconds',0):.1f} s; neural candidate time {timing['neural_trial_seconds']:.1f} s including smoke tests and quantiles; final evaluation {timing['final_seconds']:.1f} s. Trial manifests retain individual timings and failures. The runner resumes by reusing completed trials and refuses to retune an already frozen run. No ANFIS fit or Git publication was performed.

Run python -B -m src.experiment --run {run.name} --phase all from the project root with the existing venv. See [preflight](preflight.json), [freeze manifest](freeze.json), [validation metrics](validation_common_metrics.csv), [test metrics](test_common_metrics.csv), [test coverage](test_coverage.csv), [flood event timing](test_flood_events.csv), [warning timing plot](test_warning_timing.png), and [test intervals](test_quantile_metrics.json). One-step classical filtering uses the [statsmodels predicted information set](https://www.statsmodels.org/stable/generated/statsmodels.tsa.statespace.sarimax.SARIMAXResults.get_prediction.html).
"""
    (run/'research_report.md').write_text(contents,encoding='utf8')


def evaluate(stage,bounds,run):
    if (run/'final_complete.json').exists():
        return
    torch.set_num_threads(2)
    clock=perf_counter()
    freeze=freeze_protocol(run,stage,bounds)
    predictions={}
    for name,lag in BASELINES.items():
        predictions[name]=slice_predictions(stage,bounds,'test',lag)
    for name,row in freeze['classical'].items():
        fit=SARIMAXResults.load(run/f"classical_trials/{row['name']}.pkl")
        predictions[name]=classical_predict(fit,stage,bounds,'test',row['lag'])
    for name,row in freeze['neural'].items():
        predictions[name]=predict_checkpoint(run/f"neural_trials/{row['trial']}.pt",row['config'],
                                             name,stage,bounds,'test',freeze['scaler'])
    lo,hi=bounds['test']
    expected=hi-lo
    coverage=[]
    for name,pred in predictions.items():
        result=save_predictions(run,stage,pred,name,'test')
        coverage.append({'model':name,'eligible_pairs':result['n'],'possible_targets':expected,
                         'coverage_fraction':result['n']/expected})
    common=pd.concat(predictions,axis=1).dropna()
    actual=stage.loc[common.index]
    common=common.loc[actual.notna()]
    actual=actual.loc[common.index]
    comparisons=pd.DataFrame([dict(model=name,**stats(actual,common[name]))
                              for name in common.columns]).sort_values('RMSE_ft')
    comparison_val=validation_table(run,list(predictions))
    comparison_val.to_csv(run/'validation_common_metrics.csv',index=False)
    comparisons.to_csv(run/'test_common_metrics.csv',index=False)
    lower,upper=[predict_checkpoint(run/f"neural_trials/{row['trial']}.pt",row['config'],'lstm',
                                    stage,bounds,'test',freeze['scaler'])
                 for row in (freeze['quantile']['lower'],freeze['quantile']['upper'])]
    interval=save_interval(stage,lower,upper,run,'test')
    coverage.append({'model':'quantile_interval','eligible_pairs':interval['n'],'possible_targets':expected,
                     'coverage_fraction':interval['n']/expected})
    pd.DataFrame(coverage).to_csv(run/'test_coverage.csv',index=False)
    hourly,events,alarms,rises=[],[],[],[]
    threshold=freeze['rapid_rise_cutoff_ft_per_hour_training_99th_percentile']
    previous=stage.shift().loc[common.index]
    fast=(actual-previous).ge(threshold)
    for name in common:
        pred=common[name]
        scores,rows,false=warning_metrics(actual,pred,name)
        hourly.append(scores)
        events+=rows
        alarms+=false
        errors=pred[fast]-actual[fast]
        rises.append({'model':name,'rapid_rise_hours':int(fast.sum()),
                      'underestimated_fraction':float((errors<0).mean()),
                      'mean_error_ft':errors.mean(),'MAE_ft':errors.abs().mean()})
    hourly=pd.DataFrame(hourly)
    events=pd.DataFrame(events)
    hourly.to_csv(run/'test_flood_hourly.csv',index=False)
    events.to_csv(run/'test_flood_events.csv',index=False)
    pd.DataFrame(alarms).to_csv(run/'test_false_alarm_episodes.csv',index=False)
    pd.DataFrame(rises).to_csv(run/'test_rapid_rises.csv',index=False)
    hydrograph(actual,{name:common[name] for name in comparisons.model.head(3)},run/'test_common_hydrograph.png')
    flood_plot(actual,common,comparison_val,run)
    timing={'classical_trial_seconds':read_json(run/'classical_timing.json')['total_candidate_seconds'],
            'neural_trial_seconds':sum(read_json(p)['seconds'] for p in (run/'neural_trials').glob('*.json')),
            'final_seconds':perf_counter()-clock}
    write_json(run/'timing.json',timing)
    report(run,freeze,comparison_val,comparisons,hourly,events,interval,timing)
    write_json(run/'final_complete.json',{'common_test_hours':len(common),'model_count':len(predictions),
                                         'test_flood_events':len(episodes(actual.ge(7))),**timing})
