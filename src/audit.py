import json, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf, adfuller, kpss
from statsmodels.tsa.seasonal import STL
from . import config as C
from .utils import read_hourly,splits

def main():
    df=read_hourly(); s=df.stage_ft; sp=splits(len(s)); tr=s.iloc[:sp['train'][1]]
    plt.figure(figsize=(12,4));plt.plot(s.index,s,label='stage',lw=.8)
    plt.axhline(C.MINOR_STAGE,ls='--',label='7 ft flood threshold');plt.ylabel('Stage (ft)');plt.legend()
    plt.tight_layout();plt.savefig(C.FIGURES/'observed_stage.png',dpi=160);plt.close()
    clean=tr.dropna()
    if len(clean)<100:raise SystemExit('Insufficient nonmissing training observations.')
    # Stationarity diagnostic only; missing hours alter lag spacing, so see report caveat.
    adf=adfuller(clean,autolag='AIC');
    with warnings.catch_warnings():
        warnings.simplefilter('ignore'); kp=kpss(clean,nlags='auto')
    dx=clean.diff().dropna()
    adf_d=adfuller(dx,autolag='AIC')
    report={'train_n':int(len(tr)),'train_nonmissing':int(len(clean)),
            'ADF_level_p':float(adf[1]),'KPSS_level_p':float(kp[1]),'ADF_diff1_p':float(adf_d[1]),
            'note':'ADF and KPSS on dropna are only illustrative if hourly gaps exist.'}
    with open(C.METRICS/'audit_tests.json','w') as f:json.dump(report,f,indent=2)
    lags=min(72,len(clean)//5); ac=acf(clean,nlags=lags)
    plt.figure(figsize=(10,3));plt.stem(np.arange(len(ac)),ac);plt.xlabel('Lag (hours only if no missing timestamps)');plt.ylabel('ACF')
    plt.tight_layout();plt.savefig(C.FIGURES/'training_acf.png',dpi=150);plt.close()
    if tr.isna().mean()<0.001 and len(tr)>100:
        fig=STL(tr,period=24,robust=True).fit().plot();fig.set_size_inches(11,7)
        fig.tight_layout();fig.savefig(C.FIGURES/'training_stl_24h.png',dpi=150);plt.close(fig)
    else: print('SKIPPED STL: missing hours; choose a continuous block first.')
    print(json.dumps(report,indent=2))
    print('CHECK: ACF, decomposition and stationarity are diagnostics, NOT proof of forecasting skill.')
if __name__=='__main__':main()
