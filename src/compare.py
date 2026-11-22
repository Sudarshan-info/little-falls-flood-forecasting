"""Compare every forecast on the SAME timestamp rows, not different missing-data subsets."""
import pandas as pd
from . import config as C
from .utils import stats

def main():
    files=sorted(C.METRICS.glob('*_preds.csv'))
    keep=[]
    for f in files:
        if f.stem in ('quantile_preds','sau_binary_preds'):continue
        d=pd.read_csv(f)
        if {'time_utc','actual','predicted'}.issubset(d.columns):
            keep.append((f.stem.replace('_preds',''),d[['time_utc','actual','predicted']].copy()))
    if len(keep)<2:raise SystemExit('Need at least two point forecast files, e.g. baseline + ARIMA.')
    common=None
    for name,d in keep:
        q=d[['time_utc','predicted']].rename(columns={'predicted':name})
        common=q if common is None else common.merge(q,on='time_utc',how='inner')
    # Get actual measurement series from first model after alignment
    common=common.merge(keep[0][1][['time_utc','actual']],on='time_utc',how='inner')
    common=common.dropna()
    if len(common)<30:raise SystemExit('Too few shared timestamps across models; inspect missing hours.')
    rows=[]
    for name,_ in keep:rows.append({'model':name,**stats(common.actual,common[name])})
    out=pd.DataFrame(rows).sort_values('RMSE_ft');out.to_csv(C.METRICS/'model_comparison_common.csv',index=False)
    print('Common evaluated timestamps:',len(common));print(out.to_string(index=False))
    print('STOP: test rows match; explain model selection by validation, not test peeking.')
if __name__=='__main__':main()
