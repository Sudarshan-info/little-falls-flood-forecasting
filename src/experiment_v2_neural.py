"""Neural and interval controls using an inner chronological stopping slice."""

import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from .experiment import read_json
from .experiment_v2 import YEARS, cached, save_tables, score
from .utils import complete_window_targets


CONTROL={'lag':24,'hidden':16,'layers':2,'batch':384,'lr':.002,'epochs':16,
         'cap':18000,'sampling':'uniform','features':False,'decay':0.}
CONFIGS={
    'control16':CONTROL,
    'longer36':{**CONTROL,'epochs':36},
    'larger48':{**CONTROL,'lag':48,'hidden':32,'layers':1,'batch':512,'lr':.001,'epochs':26,'decay':1e-4},
}
PILOT={**CONTROL,'epochs':10,'cap':6000}
LSTM_PILOTS={
    'pilot_control':PILOT,
    'pilot_larger48':{**PILOT,'lag':48,'hidden':32,'layers':1,'batch':512,'lr':.001,'decay':1e-4},
    'pilot_coverage24':{**PILOT,'cap':9000},
    'pilot_rise24':{**PILOT,'sampling':'past_rise'},
    'pilot_features24':{**PILOT,'features':True},
}


class ResidualNet(nn.Module):
    def __init__(self,cell,cfg,interval):
        super().__init__()
        recurrent=nn.RNN if cell=='rnn' else nn.LSTM
        self.cell=recurrent(4 if cfg['features'] else 1,cfg['hidden'],
                            num_layers=cfg['layers'],batch_first=True)
        self.interval=interval
        self.head=nn.Linear(cfg['hidden'],2 if interval else 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        if interval:
            with torch.no_grad():
                self.head.bias[:] = (torch.tensor([-.012,.012]) if interval=='free'
                                     else torch.tensor([0.,-4.4]))

    def forward(self,x):
        history,_=self.cell(x)
        delta=self.head(history[:,-1])
        latest=x[:,-1,:1]
        if not self.interval:
            return (delta+latest).squeeze(-1)
        if self.interval=='ordered':
            width=F.softplus(delta[:,1:2])
            delta=torch.cat((delta[:,:1]-width,delta[:,:1]+width),dim=1)
        return delta+latest


def windows(values, start, stop, cfg, seed, training=False):
    lag=cfg['lag']
    history=values[:stop]
    pos=np.flatnonzero(complete_window_targets(history,lag)[start:stop])+start
    available=len(pos)
    if training and available>cfg['cap']:
        if cfg['sampling']=='past_rise':
            # Oversample eligible rapid-rise contexts using only the history available at issuance.
            recent=history[pos-1]-history[pos-4]
            high=history[pos-1]>=np.quantile(history[np.isfinite(history)],.98)
            weights=1+4*(recent>=np.quantile(recent,.95))+2*high
            pos=np.sort(np.random.default_rng(seed).choice(pos,cfg['cap'],replace=False,p=weights/weights.sum()))
        else:
            pos=pos[np.linspace(0,available-1,cfg['cap'],dtype=int)]
    x=np.lib.stride_tricks.sliding_window_view(history,lag)[pos-lag].copy()[:,:,None]
    if cfg['features']:
        # Each feature uses only earlier time steps within that window.
        channels=[x]
        for offset in (1,3,6):
            channel=np.zeros_like(x)
            channel[:,offset:]=x[:,offset:]-x[:,:-offset]
            channels.append(channel)
        x=np.concatenate(channels,axis=2)
    return torch.from_numpy(x),torch.from_numpy(history[pos].copy()),pos,available


def lossfn(pred,actual,interval):
    if not interval:
        return F.mse_loss(pred,actual)
    residual=actual[:,None]-pred
    q=pred.new_tensor([.025,.975])
    return torch.maximum(q*residual,(q-1)*residual).mean()


def infer(net,x):
    net.eval()
    with torch.no_grad():
        return torch.cat([net(chunk) for chunk in x.split(512)])


def neural_fit(stage,fold,cell,cfg,seed,path,interval=None):
    year,lo,hi,rise=fold
    inner=lo-90*24
    training=stage.iloc[:inner]
    mean,scale=training.mean(),training.std(ddof=0)
    values=((stage.iloc[:hi].to_numpy()-mean)/scale).astype('float32')
    x,y,pos,possible=windows(values,0,inner,cfg,seed,training=True)
    vx,vy,_,_=windows(values,inner,lo,cfg,seed)
    torch.manual_seed(seed)
    rng=torch.Generator().manual_seed(seed)
    net=ResidualNet(cell,cfg,interval)
    opt=torch.optim.Adam(net.parameters(),lr=cfg['lr'],weight_decay=cfg['decay'])
    log=[]
    best=float('inf')
    for epoch in range(1,cfg['epochs']+1):
        net.train()
        total=0.
        for idx in torch.randperm(len(y),generator=rng).split(cfg['batch']):
            opt.zero_grad()
            loss=lossfn(net(x[idx]),y[idx],interval)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(),1)
            opt.step()
            total+=float(loss.detach())*len(idx)
        val=float(lossfn(infer(net,vx),vy,interval))
        if not math.isfinite(val):
            raise ValueError('Nonfinite stopping loss.')
        log.append({'epoch':epoch,'training_loss':total/len(y),'inner_validation_loss':val})
        if val<best-1e-8:
            best,best_epoch=val,epoch
            state={key:value.detach().clone() for key,value in net.state_dict().items()}
        if epoch-best_epoch>=10:
            break
    pd.DataFrame(log).to_csv(path.with_name(path.stem+'_loss.csv'),index=False)
    net.load_state_dict(state)
    torch.save(state,path.with_suffix('.pt'))
    ox,_,positions,_=windows(values,lo,hi,cfg,seed)
    predicted=infer(net,ox).numpy().astype(float)*scale+mean
    index=stage.index[positions]
    result={'status':'ok','best_epoch':best_epoch,'epochs_run':len(log),'optimizer':'Adam',
            'inner_start':str(stage.index[inner]),'inner_end':str(stage.index[lo-1]),
            'fit_windows':len(pos),'eligible_fit_windows':possible,'inner_valid_windows':len(vy),
            'fit_positive_targets':int(stage.iloc[pos].ge(7).sum()),'scaler_mean':mean,'scaler_scale':scale}
    if not interval:
        result['metrics'],result['events']=score(stage,pd.Series(predicted,index=index),lo,hi,rise)
        return result
    coverage={}
    actual=stage.loc[index].to_numpy()
    prior=stage.shift().loc[index].to_numpy()
    lower,upper=predicted.T
    crossed=lower>upper
    inside=(actual>=np.minimum(lower,upper))&(actual<=np.maximum(lower,upper))
    width=np.abs(upper-lower)
    for label,mask in [('overall',np.ones(len(actual),bool)),('rapid',actual-prior>=rise),('flood',actual>=7)]:
        coverage[label]={'n':int(mask.sum()),'hits':int(inside[mask].sum()),'width_sum':float(width[mask].sum()),
                         'crossings':int(crossed[mask].sum())}
    pd.DataFrame({'target_utc':index,'actual_ft':actual,'q025_ft':lower,'q975_ft':upper}).to_csv(
        path.with_name(path.stem+'_interval.csv'),index=False)
    return result|{'interval_metrics':coverage}


def run_neural(stage,folds,run):
    torch.set_num_threads(2)
    refs=[score(stage,stage.iloc[:hi].shift(),lo,hi,rise)[0] for _,lo,hi,rise in folds]
    ref=math.sqrt(sum(r['normal_sse'] for r in refs)/sum(r['normal_n'] for r in refs))
    selected={}
    for cell in ('rnn','lstm'):
        rows=[]
        names=('control16','longer36','larger48') if cell=='rnn' else (
            'control16','longer36',*LSTM_PILOTS)
        for name in names:
            cfg=LSTM_PILOTS[name] if name in LSTM_PILOTS else CONFIGS[name]
            for fold in folds:
                if cell=='lstm' and name not in ('control16','longer36') and fold[0] not in (2021,2023):
                    continue
                pilot=run/'trials'/cell/f'{name}_2018.json'
                prune=(cell=='rnn' and name=='larger48' and fold[0]>2018 and pilot.exists()
                       and read_json(pilot)['seconds']>40)
                rows.append(cached(run,cell,name,fold[0],{'config':cfg,'seed':42},
                    lambda path:({'status':'pruned','reason':'RNN larger48 pilot exceeded 40 seconds'}
                                 if prune else neural_fit(stage,fold,cell,cfg,42,path))))
        ranking=save_tables(run,cell,rows,ref)
        eligible=ranking[ranking.eligible]
        if not len(eligible):
            continue
        winner=eligible.iloc[0].candidate
        selected[cell]=winner
    for cell,winner in selected.items():
        seeds=[]
        for seed in (42,43,44):
            name=f'{winner}_pilot_seed{seed}'
            for fold in folds:
                if fold[0]!=2023:
                    continue
                cfg={**CONFIGS[winner],'epochs':10,'cap':6000}
                seeds.append(cached(run,cell,name,fold[0],{'config':cfg,'seed':seed},
                    lambda path:neural_fit(stage,fold,cell,cfg,seed,path)))
        save_tables(run,f'{cell}_seeds',seeds,ref)
    if 'lstm' not in selected:
        return
    from .experiment import write_json
    write_json(run/'neural_selected.json',{cell:{'name':winner,'settings':CONFIGS[winner]}
                                         for cell,winner in selected.items()})
    cfg={**CONFIGS[selected['lstm']],'epochs':10,'cap':6000}
    rows=[]
    for interval in ('free','ordered'):
        for fold in folds:
            if fold[0] not in (2018,2021,2023):
                continue
            rows.append(cached(run,'interval',interval,fold[0],{'config':cfg,'seed':42,'bounds':interval},
                lambda path:neural_fit(stage,fold,'lstm',cfg,42,path,interval=interval)))
    rows=[r for r in rows if r['status']=='ok']
    metrics=[]
    for row in rows:
        for group, m in row['interval_metrics'].items():
            metrics.append({'candidate':row['candidate'],'year':row['year'],'subset':group,**m,
                            'epoch':row['best_epoch'],'seconds':row['seconds']})
    table=pd.DataFrame(metrics)
    if table.empty:
        return
    for keys,file in [(['candidate','year','subset'],'interval_folds.csv'),
                      (['candidate','subset'],'interval_summary.csv')]:
        totals=table.groupby(keys,as_index=False)[['n','hits','width_sum','crossings']].sum()
        totals['PICP']=totals.hits/totals.n
        totals['MPIW_ft']=totals.width_sum/totals.n
        totals['crossing_fraction']=totals.crossings/totals.n
        totals.to_csv(run/file,index=False)


def make_report(run):
    # Report generation consumes saved validation artifacts only.
    fragments=[]
    best_rows=[]
    comparisons=[]
    conclusions=[]
    seed_text=[]
    paired_neural=[]
    v1=pd.read_csv(run.parents[2]/'docs'/'research_v1'/'validation_common_metrics.csv').set_index('model')
    for family in ('baseline','classical','rnn','lstm'):
        path=run/f'{family}_summary.csv'
        if not path.exists():
            continue
        table=pd.read_csv(path)
        top=table[table.eligible].head(1)
        if family=='classical':
            ar=table.candidate.str.match(r'^\d+_0_0_')
            top=pd.concat([table[table.eligible&ar].head(1).assign(reference='ar'),
                           table[table.eligible&~ar].head(1).assign(reference='arima')])
            eligible_pair=table[table.ok_folds.eq(len(YEARS))]
            pair=eligible_pair.assign(order=eligible_pair.candidate.str.split('_cap').str[0],
                                     cap=eligible_pair.candidate.str.split('_cap').str[1])
            pair=pair.pivot(index='order',columns='cap',values='overall_rmse').dropna()
            pair['long_minus_short_rmse_ft']=pair['32000']-pair['8000']
            pair.to_csv(run/'classical_history_comparison.csv')
            conclusions.append(f"- Longer classical history lowered pooled RMSE for {int(pair.long_minus_short_rmse_ft.lt(0).sum())} of {len(pair)} paired orders. See classical_history_comparison.csv for pair differences.")
        else:
            top=top.assign(reference=('persistence' if family=='baseline' else family))
        detail=pd.read_csv(run/f'{family}_folds.csv')
        for row in top.itertuples():
            yr=detail[detail.candidate.eq(row.candidate)&detail.year.eq(2023)].iloc[0]
            original=v1.loc[row.reference]
            comparisons.append({'model_family':row.reference,'v2_candidate':row.candidate,
                                'v1_2023_rmse_ft':original.RMSE_ft,'v1_2023_mae_ft':original.MAE_ft,
                                'v2_2023_n':yr.overall_n,'v2_2023_rmse_ft':yr.overall_rmse,
                                'v2_2023_mae_ft':yr.overall_mae})
        if family in ('lstm','rnn'):
            ref=table.set_index('candidate')
            for experiment,control in [('longer36','control16'),('pilot_coverage24','pilot_control'),
                                       ('pilot_rise24','pilot_control'),('pilot_features24','pilot_control'),
                                       ('larger48','control16'),('pilot_larger48','pilot_control')]:
                if {experiment,control}<=set(ref.index):
                    x=detail[detail.candidate.eq(experiment)&detail.status.eq('ok')].set_index('year')
                    y=detail[detail.candidate.eq(control)&detail.status.eq('ok')].set_index('year')
                    common=x.index.intersection(y.index)
                    if common.empty:
                        continue
                    x,y=x.loc[common],y.loc[common]
                    delta={}
                    for subset in ('overall','rapid','flood'):
                        key=f'{subset}_n'
                        if not np.array_equal(x[key],y[key]):
                            raise ValueError('Paired controls do not share the scoring grid.')
                        n=x[key].sum()
                        delta[subset]=(math.sqrt(x[f'{subset}_sse'].sum()/n)
                                       -math.sqrt(y[f'{subset}_sse'].sum()/n)) if n else np.nan
                    paired_neural.append({'family':family,'candidate':experiment,'control':control,
                                          'years':','.join(map(str,common)),**delta})
                    conclusions.append(f"- {family.upper()} {experiment} vs {control} on {len(common)} matched folds: change in overall / rapid-rise / flood RMSE was {delta['overall']:+.5f} / {delta['rapid']:+.5f} / {delta['flood']:+.5f} ft.")
            seed_path=run/f'{family}_seeds_summary.csv'
            if seed_path.exists():
                for s in pd.read_csv(seed_path).itertuples():
                    seed_text.append(f'- {family.upper()} {s.candidate}: pooled RMSE {s.overall_rmse:.5f} ft, rapid RMSE {s.rapid_rmse:.5f} ft, {s.ok_folds} successful folds.')
            if len(top):
                winner=top.iloc[0].candidate
                fig,axes=plt.subplots(1,len(YEARS),figsize=(15,3.4),layout='constrained')
                for year,ax in zip(YEARS,axes):
                    losses=pd.read_csv(run/'trials'/family/f'{winner}_{year}_loss.csv')
                    ax.plot(losses.epoch,losses.training_loss,label='Train')
                    ax.plot(losses.epoch,losses.inner_validation_loss,label='Inner')
                    ax.set(title=str(year),xlabel='Epoch')
                axes[0].set_ylabel('Scaled MSE')
                axes[-1].legend(fontsize=8)
                fig.suptitle(f'{family.upper()} winning candidate {winner}')
                fig.savefig(run/f'{family}_winning_loss.png',dpi=140)
                plt.close(fig)
        best_rows.append(top.assign(family=family))
    finalists=pd.concat(best_rows,ignore_index=True)
    finalists.to_csv(run/'validation_finalists.csv',index=False)
    pd.DataFrame(paired_neural).to_csv(run/'paired_neural_controls.csv',index=False)
    from .experiment import write_json
    write_json(run/'classical_selected.json',{
        row.reference:{'candidate':row.candidate,'order':list(map(int,row.candidate.split('_')[:3])),
                       'training_cap_hours':int(row.candidate.split('_cap')[1])}
        for row in finalists[finalists.family.eq('classical')].itertuples()})
    comparison=pd.DataFrame(comparisons)
    comparison.to_csv(run/'matched_2023_v1_comparison.csv',index=False)
    lines_2023=['| Family | V2 candidate | V1 RMSE | V2 RMSE | V1 MAE | V2 MAE |',
                '|---|---|---:|---:|---:|---:|']
    for row in comparison.itertuples():
        lines_2023.append(f'| {row.model_family} | {row.v2_candidate} | {row.v1_2023_rmse_ft:.5f} | {row.v2_2023_rmse_ft:.5f} | {row.v1_2023_mae_ft:.5f} | {row.v2_2023_mae_ft:.5f} |')
    intervals=[]
    interval_2023=[]
    if (run/'interval_summary.csv').exists():
        for row in pd.read_csv(run/'interval_summary.csv').itertuples():
            intervals.append(f'| {row.candidate} | {row.subset} | {row.n} | {row.PICP:.2%} | {row.MPIW_ft:.4f} | {row.crossings} |')
        reference=read_json(run.parents[2]/'docs'/'research_v1'/'validation_quantile_metrics.json')
        interval_2023.append(f"- Saved V1 2023 intervals: coverage {reference['PICP']:.2%}, width {reference['MPIW_ft']:.4f} ft, {reference['crossings']} raw crossings.")
        by_year=pd.read_csv(run/'interval_folds.csv')
        for row in by_year[by_year.year.eq(2023)&by_year.subset.eq('overall')].itertuples():
            interval_2023.append(f'- V2 {row.candidate} in 2023: coverage {row.PICP:.2%}, width {row.MPIW_ft:.4f} ft, {row.crossings} raw crossings.')
    event_frames=[]
    onset_text=[]
    for row in finalists.itertuples():
        dates=pd.read_csv(run/f'{row.family}_events.csv')
        dates=dates[dates.candidate.eq(row.candidate)].assign(family=row.family)
        if len(dates):
            event_frames.append(dates)
            onset_text.append(f"- {row.family} {row.candidate}: {int(dates.first_hit_delay_h.eq(0).sum())}/{len(dates)} events hit at the hourly onset target.")
    if event_frames:
        pd.concat(event_frames).to_csv(run/'finalists_flood_events.csv',index=False)
    timing=[]
    for folder in sorted((run/'trials').iterdir()):
        rows=[read_json(f) for f in folder.glob('*.json')]
        timing.append({'family':folder.name,'attempted_folds':len(rows),
                       'fit_and_evaluation_seconds':sum(r['seconds'] for r in rows),
                       'success':sum(r['status']=='ok' for r in rows),
                       'nonconverged':sum(r['status']=='nonconverged' for r in rows),
                       'failed':sum(r['status']=='failed' for r in rows),
                       'pruned':sum(r['status']=='pruned' for r in rows)})
    pd.DataFrame(timing).to_csv(run/'runtime.csv',index=False)
    for row in finalists.itertuples():
        fragments.append(f'| {row.family} | {row.candidate} | {row.overall_rmse:.5f} | {row.overall_mae:.5f} | '
                         f'{row.rapid_rmse:.5f} | {row.worst_year_rmse:.5f} | {row.events_detected}/{row.events} |')
    fig,ax=plt.subplots(figsize=(10,4),layout='constrained')
    for row in finalists.itertuples():
        detail=pd.read_csv(run/f'{row.family}_folds.csv')
        series=detail[detail.candidate.eq(row.candidate)]
        ax.plot(series.year,series.overall_rmse,marker='o',label=f'{row.family} {row.candidate}')
    ax.set(xlabel='Validation year',ylabel='RMSE (ft)',title='Expanding-window historical validation')
    ax.set_xticks(YEARS)
    ax.legend(fontsize=8)
    fig.savefig(run/'fold_comparison.png',dpi=140)
    plt.close(fig)
    expected=('classical_summary.csv','rnn_summary.csv','lstm_summary.csv',
              'rnn_seeds_summary.csv','lstm_seeds_summary.csv','interval_summary.csv')
    pending=[name for name in expected if not (run/name).exists()]
    status='Partial. Pending artifacts ' + ', '.join(pending) if pending else 'Completed bounded historical search.'
    selected_point=finalists[~finalists.family.eq('baseline')]
    stable=selected_point.loc[selected_point.worst_year_rmse.idxmin(),'candidate'] if len(selected_point) else 'pending'
    low_sd=selected_point.loc[selected_point.year_rmse_sd.idxmin(),'candidate'] if len(selected_point) else 'pending'
    report=f"""# Little Falls V2 historical optimization

{status}

Selection uses pooled hourly RMSE across 2014, 2018, 2021, 2022 and 2023 with a cap of 105% of persistence's normal-stage RMSE. All point comparisons score the same targets, requiring a complete target and preceding 48 hours. Each fold uses earlier observations only. No 2024–2025 records entered the modeling frame. See folds.csv for absolute cutoffs and sample counts.

| Family | Candidate | RMSE (ft) | MAE (ft) | Rapid-rise RMSE (ft) | Worst-year RMSE (ft) | Events detected |
|---|---|---:|---:|---:|---:|---:|
{chr(10).join(fragments)}

Among the reported finalists, the lowest worst-year RMSE belongs to {stable}. The smallest standard deviation of annual RMSE belongs to {low_sd}. See validation_finalists.csv for both measures.

Classical fits use the latest contiguous block before each fold, capped at 8,000 or 32,000 hours. Earlier outages limited both histories to 5,954, 7,148 and 2,600 hours for the 2014, 2018 and 2021 folds. The 2022 and 2023 folds are the ones where the higher cap adds fitting observations. The longer 2023 block has 20,120 observations and 175 flood hours, compared with 57 flood hours under the 8,000-hour cap. ARIMA uses past-only state-space filtering with fixed fitted parameters. Its memory-saving implementation reproduced all three saved V1 classical 2023 forecast series exactly in a check without refitting. Nonconverged fits are recorded and ineligible.

Each neural fold reserves the last 90 days before the outer year for early stopping. The scaler and training samples exclude that inner slice. The LSTM epoch controls run on all five folds with 18,000 sampled windows. After benchmarking compute cost, larger capacity, denser sampling (6,000 versus 9,000 windows), preferential sampling of rising or elevated past stages, and past-change features were restricted to 10-epoch pilots on 2021 and 2023. Those variants are compared with the same-resource control and are excluded from the five-fold rankings. The larger model changes several parameters together, so it measures a configuration effect rather than a hidden-unit effect. Main training is capped at 36 epochs and uses patience 10. Seed 42 selects configurations. Seeds 42, 43 and 44 were compared on 2023 with the winning architectures and a shared 6,000-window / 10-epoch budget. This is a pilot sensitivity check, and full-budget seed stability remains unmeasured. Trial JSON files record the exact settings, sample sizes and runtime, and the accompanying loss CSVs preserve each training curve.

## Evidence from matched controls

Negative changes favor the first configuration.

{chr(10).join(conclusions)}

{chr(10).join(seed_text)}

## Comparison with the saved V1 2023 validation reference

{chr(10).join(lines_2023)}

This is a same-year comparison of saved forecasts and models selected under different protocols. V1 used 2023 directly for stopping and selection. V2 used a pre-2023 inner slice for stopping and includes 2023 among its outer selection folds. The V2 controls share the V1-like architecture or order where indicated, but do not recreate the V1 training protocol for every historical fold.

## Quantile intervals

The interval comparison uses two independently emitted bounds versus a joint noncrossing parameterization with positive softplus width on 2018, 2021 and 2023. Both use the same joint pinball loss, 6,000-window / 10-epoch pilot budget, and validation protocol. Coverage for the free bounds uses ordered endpoints and separately counts raw crossings. No calibration was fitted. Empirical coverage does not guarantee future coverage.

| Candidate | Subset | Hours | PICP | Mean width (ft) | Raw crossings |
|---|---|---:|---:|---:|---:|
{chr(10).join(intervals)}

{chr(10).join(interval_2023)}

Inspect the rapid-rise rows before choosing an interval model. The ordered architecture guarantees coherent endpoints, and the free pilot also happened to have zero crossings. Neither pilot covered rapid increases well enough to support warning use. The lower rapid-rise coverage and narrower bounds for the ordered pilot are a calibration trade-off in this bounded experiment.

## Event onset timing

{chr(10).join(onset_text)}

This is historical validation for model development. Event detection requires overlapping threshold-positive forecast and observed targets. Hours within an event are dependent, and the folds contain few events. Use a locked prospective or otherwise unexamined period before a new independent test. V1's published test remains a historical reference.

Resume with python -B -m src.experiment_v2 --phase all. Completed trial manifests are reused; candidate failures stay recorded. runtime.csv gives compute totals and failures. CPU Torch threads were limited to two. The grids were deliberately bounded: lookbacks 24 and 48 hours and hidden sizes 16 and 32 were investigated, leaving the remaining grid values untested.
"""
    (run/'research_report.md').write_text(report,encoding='utf-8')
