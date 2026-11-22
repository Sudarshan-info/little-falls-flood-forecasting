"""Validation-only recurrent and quantile searches for the canonical experiment."""

from time import perf_counter
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch

from .experiment import hydrograph, read_json, save_predictions, stats, write_json
from .neural import StageNet
from .utils import complete_window_targets


SEARCH = [
    {'lookback':12,'hidden':12,'layers':1,'batch':256,'lr':.002,'epochs':16},
    {'lookback':24,'hidden':16,'layers':1,'batch':256,'lr':.001,'epochs':18},
    {'lookback':36,'hidden':24,'layers':1,'batch':384,'lr':.0015,'epochs':16},
    {'lookback':24,'hidden':16,'layers':2,'batch':384,'lr':.002,'epochs':16},
]
SEED = 42
CAP = 18000
PATIENCE = 7


def model_for(cell, cfg):
    net = StageNet(cell, cfg['hidden'], cfg['layers'])
    torch.nn.init.zeros_(net.linear.weight)
    torch.nn.init.zeros_(net.linear.bias)
    return net


def dataset(stage, bounds, split, cfg, scaler, cap=None):
    lo, hi = bounds[split]
    lag = cfg['lookback']
    z = ((stage.iloc[:hi].to_numpy(dtype=float)-scaler['mean'])/scaler['scale']).astype(np.float32)
    pos = np.flatnonzero(complete_window_targets(z,lag)[lo:hi])+lo
    if cap is not None and len(pos)>cap:
        pos = pos[np.linspace(0,len(pos)-1,cap,dtype=int)]
    if not len(pos):
        raise ValueError(f'No complete windows in {split}.')
    windows = np.lib.stride_tricks.sliding_window_view(z,lag)[pos-lag]
    X = torch.from_numpy(windows.copy())[:,:,None]
    y = torch.from_numpy(z[pos].copy())
    return X, y, stage.index[pos]


def loss_at(pred, target, quantile):
    err = target-pred
    if quantile is None:
        return err.square().mean()
    return torch.maximum(quantile*err,(quantile-1)*err).mean()


def infer(net, X):
    net.eval()
    with torch.no_grad():
        return torch.cat([net(batch)+batch[:,-1,0] for batch in X.split(512)])


def predict_checkpoint(path, cfg, cell, stage, bounds, split, scaler):
    net = model_for(cell,cfg)
    net.load_state_dict(torch.load(path,map_location='cpu',weights_only=True))
    X, _, index = dataset(stage,bounds,split,cfg,scaler)
    return pd.Series(infer(net,X).numpy().astype(float)*scaler['scale']+scaler['mean'],index=index)


def trial(stage, bounds, scaler, cell, cfg, folder, key, cap, quantile=None, seed=SEED):
    manifest = folder/f'{key}.json'
    if manifest.exists():
        return read_json(manifest)
    start = perf_counter()
    result = {'trial':key,'cell':cell,'config':cfg,'seed':seed,'quantile':quantile}
    try:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        net = model_for(cell,cfg)
        optimizer = torch.optim.Adam(net.parameters(),lr=cfg['lr'])
        X,y,_ = dataset(stage,bounds,'train',cfg,scaler,cap=cap)
        VX,VY,index = dataset(stage,bounds,'val',cfg,scaler)
        result.update(train_windows=len(y),validation_windows=len(VY))
        scores = []
        best = float('inf')
        checkpoint = folder/f'{key}.pt'
        for ep in range(1,cfg['epochs']+1):
            ep_start = perf_counter()
            net.train()
            loss_sum = 0.
            for idx in torch.randperm(len(y)).split(cfg['batch']):
                batch = X[idx]
                optimizer.zero_grad()
                loss = loss_at(net(batch)+batch[:,-1,0],y[idx],quantile)
                if not torch.isfinite(loss):
                    raise ValueError('Non-finite training loss.')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(),1.)
                optimizer.step()
                loss_sum += loss.item()*len(idx)
            prediction = infer(net,VX)
            val = loss_at(prediction,VY,quantile).item()
            if not np.isfinite(val):
                raise ValueError('Non-finite validation loss.')
            scores.append({'epoch':ep,'train_loss':loss_sum/len(y),'validation_loss':val,
                           'seconds':perf_counter()-ep_start})
            if val<best-1e-8:
                best=val
                result['best_epoch']=ep
                torch.save(net.state_dict(),checkpoint.with_suffix('.tmp'))
                checkpoint.with_suffix('.tmp').replace(checkpoint)
            if ep-result['best_epoch']>=PATIENCE:
                break
        pd.DataFrame(scores).to_csv(folder/f'{key}_loss.csv',index=False)
        pred = predict_checkpoint(checkpoint,cfg,cell,stage,bounds,'val',scaler)
        result.update(status='ok',epochs_run=len(scores),selection_loss=best,
                      **stats(stage.loc[index],pred))
    except Exception as error:
        result.update(status='failed',error=f'{type(error).__name__}: {error}')
    result['seconds'] = perf_counter()-start
    write_json(manifest,result)
    print(key, result['status'], result.get('RMSE_ft'), round(result['seconds'],2), flush=True)
    return result


def interval_metrics(actual, lower, upper, previous):
    width = upper-lower
    crossed = width.lt(0)
    left, right = np.minimum(lower,upper),np.maximum(lower,upper)
    included = actual.ge(left)&actual.le(right)
    rising_flood = actual.ge(7)&actual.gt(previous)
    return {'n':len(actual),'PICP':float(included.mean()),'MPIW_ft':float((right-left).mean()),
            'crossings':int(crossed.sum()),'crossing_pct':float(100*crossed.mean()),
            'rising_flood_n':int(rising_flood.sum()),
            'rising_flood_PICP':float(included[rising_flood].mean()),
            'rising_flood_MPIW_ft':float((right-left)[rising_flood].mean())}


def save_interval(stage, lower, upper, run, split):
    index=lower.index.intersection(upper.index)
    lower,upper=lower.loc[index],upper.loc[index]
    actual=stage.loc[index]
    prior=stage.shift(1).loc[index]
    figure,axis=plt.subplots(figsize=(13,4),layout='constrained')
    center=actual.idxmax()
    selection=index[(index>=center-pd.Timedelta(hours=70))&(index<=center+pd.Timedelta(hours=130))]
    actual.loc[selection].plot(ax=axis,label='Observed',color='#26394a')
    axis.fill_between(selection,np.minimum(lower,upper).loc[selection],np.maximum(lower,upper).loc[selection],
                      color='#539db1',alpha=.36,label='Nominal 95% interval')
    axis.axhline(7,color='#9b342d',ls='--',lw=.9)
    axis.set(ylabel='Stage (ft)',xlabel='Target hour (UTC)')
    axis.legend()
    figure.savefig(run/f'{split}_quantile_interval.png',dpi=150)
    plt.close(figure)
    pd.DataFrame({'issued_utc':index-pd.Timedelta(hours=1),'target_utc':index,'actual_ft':actual.to_numpy(),
                  'q025_ft':lower.to_numpy(),'q975_ft':upper.to_numpy()}).to_csv(
        run/f'{split}_quantile_predictions.csv',index=False)
    result=interval_metrics(actual,lower,upper,prior)
    write_json(run/f'{split}_quantile_metrics.json',result)
    return result


def run_neural(stage,bounds,run):
    torch.set_num_threads(2)
    if (run/'freeze.json').exists():
        return
    selected_path=run/'neural_selected.json'
    folder=run/'neural_trials'
    folder.mkdir(exist_ok=True)
    train=stage.iloc[:bounds['train'][1]].dropna()
    fitted=StandardScaler().fit(train.to_numpy().reshape(-1,1))
    scaler={'mean':float(fitted.mean_[0]),'scale':float(fitted.scale_[0]),'n':len(train)}
    write_json(run/'training_scaler.json',scaler)
    selected={}
    for cell in ('rnn','lstm'):
        smoke=[trial(stage,bounds,scaler,cell,{**cfg,'epochs':5},folder,
                     f'{cell}_smoke{i}',2048) for i,cfg in enumerate(SEARCH[:2],1)]
        if any(row['status']!='ok' for row in smoke):
            raise RuntimeError(f'{cell} smoke testing failed. Inspect trial manifests.')
        trials=[trial(stage,bounds,scaler,cell,cfg,folder,f'{cell}_{i}',CAP)
                for i,cfg in enumerate(SEARCH,1)]
        pd.json_normalize(smoke+trials).to_csv(run/f'{cell}_search.csv',index=False)
        candidates=[row for row in trials if row['status']=='ok']
        if not candidates:
            raise RuntimeError(f'No successful {cell} candidates.')
        winner=min(candidates,key=lambda row:row['RMSE_ft'])
        selected[cell]=winner
        path=folder/f"{winner['trial']}.pt"
        pred=predict_checkpoint(path,winner['config'],cell,stage,bounds,'val',scaler)
        save_predictions(run,stage,pred,cell,'validation')
        hydrograph(stage.loc[pred.index],{cell:pred},run/f'validation_{cell}.png')
        losses=pd.read_csv(folder/f"{winner['trial']}_loss.csv")
        axis=losses.plot(x='epoch',y=['train_loss','validation_loss'],ylabel='Scaled stage MSE')
        axis.axvline(winner['best_epoch'],ls='--',color='#687789')
        axis.figure.savefig(run/f'{cell}_best_loss.png',dpi=150,bbox_inches='tight')
        plt.close(axis.figure)
        write_json(selected_path,selected)

    config=selected['lstm']['config']
    repeats=[selected['lstm']]
    repeats.extend(trial(stage,bounds,scaler,'lstm',config,folder,f'lstm_repeat_seed{seed}',
                         CAP,seed=seed) for seed in (43,44))
    pd.json_normalize(repeats).to_csv(run/'lstm_seed_sensitivity.csv',index=False)
    quantiles={}
    for name,q in [('lower',.025),('upper',.975)]:
        row=trial(stage,bounds,scaler,'lstm',config,folder,f'quantile_{name}',CAP,quantile=q)
        if row['status']!='ok':
            raise RuntimeError(f'Quantile fit failed: {name}. See trial manifest.')
        quantiles[name]=row
    write_json(run/'quantile_selected.json',quantiles)
    lower,upper=[predict_checkpoint(folder/f"{row['trial']}.pt",row['config'],'lstm',
                                    stage,bounds,'val',scaler) for row in quantiles.values()]
    save_interval(stage,lower,upper,run,'validation')
