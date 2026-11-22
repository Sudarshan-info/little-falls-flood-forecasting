import json
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from . import config as C

def read_hourly():
    p = C.PROCESSED / 'hourly_stage.csv'
    if not p.exists():
        raise FileNotFoundError('Run python -m src.prepare before this step.')
    df = pd.read_csv(p, parse_dates=['time_utc'])
    df = df.set_index('time_utc').sort_index()
    return df

def splits(n):
    a = int(n * C.TRAIN_FRAC)
    b = a + int(n * C.VAL_FRAC)
    assert 0 < a < b < n
    return {'train': (0, a), 'val': (a,b), 'test': (b,n)}

def stats(y, pred):
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    mask = np.isfinite(y) & np.isfinite(pred)
    y, pred = y[mask], pred[mask]
    if len(y) == 0:
        raise ValueError('No valid actual/predicted pairs. Check missing values.')
    mape_mask = np.abs(y) > max(1e-8, 0.01 * np.nanmean(np.abs(y)))
    return {
        'n': int(len(y)),
        'RMSE_ft': float(np.sqrt(mean_squared_error(y,pred))),
        'MAE_ft': float(mean_absolute_error(y,pred)),
        'MAPE_percent_excluding_near_zero': float(np.mean(np.abs((y[mape_mask]-pred[mape_mask])/y[mape_mask]))*100) if mape_mask.any() else None,
        'R2': float(r2_score(y,pred)) if len(y)>1 else None,
        'excluded_from_MAPE': int((~mape_mask).sum()),
    }

def record(name, dates, actual, predicted, extra=None):
    df = pd.DataFrame({'time_utc': dates, 'actual': actual, 'predicted': predicted})
    if extra:
        for k,v in extra.items(): df[k]=v
    df.to_csv(C.METRICS / f'{name}_preds.csv', index=False)
    metrics = stats(actual, predicted)
    with open(C.METRICS / f'{name}_metrics.json','w') as f: json.dump(metrics,f,indent=2)
    s = df.dropna(subset=['actual','predicted'])
    tail = s.tail(min(400,len(s)))
    plt.figure(figsize=(11,4)); plt.plot(tail['time_utc'],tail['actual'],label='Observed')
    plt.plot(tail['time_utc'],tail['predicted'],label=name)
    plt.ylabel('River stage (ft, gage datum)'); plt.legend(); plt.xticks(rotation=25)
    plt.tight_layout(); plt.savefig(C.FIGURES / f'{name}_actual_vs_pred.png',dpi=150); plt.close()
    print(name, json.dumps(metrics,indent=2))
    return metrics

def stage_class(v):
    if not np.isfinite(v): return 'missing'
    if v >= C.MAJOR_STAGE: return 'major'
    if v >= C.MODERATE_STAGE: return 'moderate'
    if v >= C.MINOR_STAGE: return 'minor'
    if v >= C.ACTION_STAGE: return 'action'
    return 'below action'

def binary_scores(y, pred):
    from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
    y, pred = np.asarray(y).astype(int),np.asarray(pred).astype(int)
    tn,fp,fn,tp = confusion_matrix(y,pred,labels=[0,1]).ravel()
    return {'n':int(len(y)),'positives':int(y.sum()),'TP':int(tp),'FP':int(fp),'FN':int(fn),'TN':int(tn),
            'precision':float(precision_score(y,pred,zero_division=0)),
            'recall':float(recall_score(y,pred,zero_division=0)),
            'F1':float(f1_score(y,pred,zero_division=0))}
