"""Two quantile LSTMs (2.5%, 97.5%) for a nominal 95% interval."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from . import config as C
from .utils import read_hourly
from .neural import prepare,train,forecast

def pinball(q):
    def loss(pred,true):
        e=true-pred
        return torch.mean(torch.maximum(q*e,(q-1)*e))
    return loss

def main():
    cp=C.METRICS/'lstm_best.json'
    if not cp.exists():raise SystemExit('Train LSTM first: python -m src.neural lstm --trials 2')
    cfg=json.load(open(cp));y=read_hourly().stage_ft
    data,sc=prepare(y,cfg['lag'])
    outputs=[]
    for q in (.025,.975):
        m,_=train(data,'lstm',cfg['hidden'],cfg['layers'],loss_fn=pinball(q),epochs=60)
        X,yt,idx=data['test'];outputs.append(forecast(m,X,sc))
    low,high=outputs
    crossed=(low>high).sum();lower=np.minimum(low,high);upper=np.maximum(low,high)
    _,_,idx=data['test'];actual=y.to_numpy()[idx]
    coverage=float(np.mean((actual>=lower)&(actual<=upper)))
    width=float(np.mean(upper-lower))
    o={'PICP':coverage,'MPIW_ft':width,'nominal_coverage':.95,'crossings':int(crossed), 'n':int(len(actual))}
    json.dump(o,open(C.METRICS/'quantile_metrics.json','w'),indent=2)
    pd.DataFrame({'time_utc':y.index[idx],'actual':actual,'lower':lower,'upper':upper}).to_csv(C.METRICS/'quantile_preds.csv',index=False)
    n=min(len(idx),500);j=np.arange(len(idx)-n,len(idx))
    plt.figure(figsize=(10,4));plt.fill_between(y.index[idx][j],lower[j],upper[j],alpha=.25,label='nominal 95% interval')
    plt.plot(y.index[idx][j],actual[j],label='actual',lw=1.0);plt.ylabel('Stage (ft)');plt.legend();plt.tight_layout()
    plt.savefig(C.FIGURES/'quantile_interval.png',dpi=150);plt.close()
    print(o,'NOTE: empirical coverage is not guaranteed; review missed peaks.')
if __name__=='__main__':main()
