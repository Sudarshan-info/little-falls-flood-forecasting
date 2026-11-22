"""One-step-ahead AR / MA / ARIMA / SARIMA using coefficients fitted to training only."""
import argparse, json, warnings
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from . import config as C
from .utils import complete_window_targets, read_hourly, forecast_splits, record

def forecast(y, order, seasonal=(0,0,0,0)):
    sp=forecast_splits(y.index); train=y.iloc[:sp['train'][1]]
    trend='c' if (order[1]==0 and seasonal[1]==0) else 'n'
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        fit=SARIMAX(train,order=order,seasonal_order=seasonal,trend=trend,
                    enforce_stationarity=False,enforce_invertibility=False).fit(disp=False,maxiter=100)
    applied = fit.apply(y, refit=False)
    predicted = applied.get_prediction(start=0, end=len(y)-1,
                                       information_set='predicted').predicted_mean
    # State-space fitting handles missing observations explicitly. Score only
    # targets with consecutive observed hours over the specified lag span.
    seasonal_span = max(seasonal[:3]) * seasonal[3]
    lookback = max(1, max(order) + seasonal_span)
    return fit, predicted.where(complete_window_targets(y, lookback))

def main():
    p=argparse.ArgumentParser()
    p.add_argument('model',choices=['ma','ar','arima','sarima'])
    p.add_argument('--p',type=int,default=2);p.add_argument('--d',type=int,default=1)
    p.add_argument('--q',type=int,default=1);p.add_argument('--s',type=int,default=24)
    a=p.parse_args(); y=read_hourly().stage_ft
    order={'ma':(0,0,a.q),'ar':(a.p,0,0),'arima':(a.p,a.d,a.q),'sarima':(a.p,a.d,a.q)}[a.model]
    seasonal=(0,1,0,a.s) if a.model=='sarima' else (0,0,0,0)
    f,pred=forecast(y,order,seasonal); t0,t1=forecast_splits(y.index)['test']
    name=a.model+'_'+''.join(str(z) for z in order)
    record(name,y.index[t0:t1],y.iloc[t0:t1].values,pred.iloc[t0:t1].values)
    print('Chosen order',order,'seasonal',seasonal,'training AIC',f.aic)
    if not f.mle_retvals.get('converged',True):print('WARNING: optimizer did not converge.')
    print('STOP: demonstrate WHY these p/d/q values were selected (ACF, ADF, validation RMSE).')
if __name__=='__main__':main()
