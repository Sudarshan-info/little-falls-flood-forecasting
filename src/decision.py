"""Compare warning policies on shared test times. Illustrative costs are user assumptions."""
import argparse
import numpy as np
import pandas as pd
from . import config as C
from .utils import binary_scores

def main():
    p=argparse.ArgumentParser();p.add_argument('--miss-cost',type=float,default=None)
    p.add_argument('--false-alarm-cost',type=float,default=None);a=p.parse_args()
    files=['persistence_preds.csv','arima_211_preds.csv','lstm_preds.csv','quantile_preds.csv']
    frames=[]
    for fname in files:
        path=C.METRICS/fname
        if not path.exists():continue
        df=pd.read_csv(path)
        col='upper' if fname=='quantile_preds.csv' else 'predicted'
        if not {'time_utc','actual',col}.issubset(df.columns):continue
        name=fname.replace('_preds.csv','')
        frames.append((name,df[['time_utc',col]].rename(columns={col:name})))
    if not frames:raise SystemExit('Run at least baselines first.')
    common=pd.read_csv(C.METRICS/'persistence_preds.csv')[['time_utc','actual']]
    for name,df in frames:common=common.merge(df,on='time_utc',how='inner')
    common=common.dropna()
    if common.empty:raise SystemExit('No shared timestamps among warning policies.')
    actual=(common.actual>=C.MINOR_STAGE).astype(int)
    rows=[]
    for name,_ in frames:
        predicted=(common[name]>=C.MINOR_STAGE).astype(int)
        m=binary_scores(actual,predicted);m['policy']=name
        if a.miss_cost is not None and a.false_alarm_cost is not None:
            m['assumed_cost_units']=m['FN']*a.miss_cost+m['FP']*a.false_alarm_cost
        rows.append(m)
    out=pd.DataFrame(rows);out.to_csv(C.METRICS/'warning_comparison.csv',index=False)
    print('Shared test hours:',len(common));print(out.to_string(index=False))
    print('WARNING: multiple flood hours during one event are not independent events.')
    print('Any costs are scenario assumptions, not independently measured savings.')
if __name__=='__main__':main()
