"""Offline programming smoke test only. THESE ARE NOT OBSERVATIONS; NEVER USE IN REPORT."""
import numpy as np,pandas as pd
from . import config as C

def main():
    rng=np.random.default_rng(11);n=2300;t=np.arange(n)
    y=3.6+.35*np.sin(t/18)+.08*np.sin(t/3)+rng.normal(0,.05,n)
    for center in range(180,n,310):
        y+=4.7*np.exp(-.5*((t-center)/26)**2)
    y[[300,301,1200]]=np.nan
    dates=pd.date_range('2024-01-01',periods=n,freq='h',tz='UTC')
    df=pd.DataFrame({'time_utc':dates,'stage_ft':y,'unit':'ft',
                     'approval_status':'synthetic','qualifier':'synthetic','time_series_id':'DEMO'})
    df.to_csv(C.RAW/'synthetic_measurements.csv',index=False)
    print('Wrote synthetic data to data/raw/synthetic_measurements.csv.')
if __name__=='__main__':main()
