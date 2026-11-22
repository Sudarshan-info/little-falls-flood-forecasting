"""Simple RNN or LSTM with chronological split and train-only scaling."""
import argparse, copy, json,random
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler
from . import config as C
from .utils import read_hourly,splits,record

class StageNet(nn.Module):
    def __init__(self, cell='lstm',hidden=16,layers=1):
        super().__init__(); cls=nn.LSTM if cell=='lstm' else nn.RNN
        self.recurrent=cls(1,hidden,num_layers=layers,batch_first=True)
        self.linear=nn.Linear(hidden,1)
    def forward(self,x):
        z,_=self.recurrent(x);return self.linear(z[:,-1,:]).squeeze(-1)

def windows(values,start,end,lag):
    idx=[];x=[];y=[]
    for t in range(max(start,lag),end):
        v=values[t-lag:t]; target=values[t]
        if np.isfinite(v).all() and np.isfinite(target):
            idx.append(t);x.append(v)
            y.append(target)
    if not x:raise ValueError('No complete windows. Check gaps or reduce lag.')
    return torch.tensor(np.array(x)[:,:,None],dtype=torch.float32),torch.tensor(y,dtype=torch.float32),np.array(idx)

def prepare(series,lag):
    s=series.to_numpy(dtype=float); sp=splits(len(s))
    scaler=StandardScaler().fit(s[:sp['train'][1]][np.isfinite(s[:sp['train'][1]])].reshape(-1,1))
    z=(s-scaler.mean_[0])/scaler.scale_[0]
    data={k:windows(z,*sp[k],lag) for k in ('train','val','test')}
    return data,scaler

def train(data, cell, hidden, layers, loss_fn=None, epochs=35, patience=5, learning_rate=.003):
    random.seed(C.SEED);np.random.seed(C.SEED);torch.manual_seed(C.SEED)
    net=StageNet(cell,hidden,layers);opt=torch.optim.Adam(net.parameters(),lr=learning_rate)
    loss_fn=loss_fn or nn.MSELoss();xtr,ytr,_=data['train'];xv,yv,_=data['val']
    best=float('inf');state=None;bad=0
    for epoch in range(epochs):
        net.train();perm=torch.randperm(len(xtr))
        for ii in range(0,len(perm),128):
            b=perm[ii:ii+128];opt.zero_grad();loss=loss_fn(net(xtr[b]),ytr[b]);loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(),1.);opt.step()
        net.eval()
        with torch.no_grad():val=float(loss_fn(net(xv),yv))
        if val<best-1e-6:best=val;state=copy.deepcopy(net.state_dict());bad=0
        else:
            bad+=1
            if bad>=patience:break
    if state is None:raise RuntimeError('No valid training state; investigate NaNs or learning rate.')
    net.load_state_dict(state)
    return net,best

def forecast(net,X,scaler):
    net.eval()
    with torch.no_grad():v=net(X).numpy()
    return v*scaler.scale_[0]+scaler.mean_[0]

def main():
    p=argparse.ArgumentParser();p.add_argument('cell',choices=['rnn','lstm']);p.add_argument('--trials',type=int,default=2)
    p.add_argument('--epochs',type=int,default=35);a=p.parse_args()
    y=read_hourly().stage_ft
    grid=[(12,12,1),(24,16,1),(48,16,1),(24,32,2),(72,32,1)]
    best=None;trial_rows=[]
    for lag,hidden,layers in grid[:a.trials]:
        data,sc=prepare(y,lag)
        model,score=train(data,a.cell,hidden,layers,epochs=a.epochs)
        trial_rows.append({'lag':lag,'hidden':hidden,'layers':layers,'validation_loss':score})
        if best is None or score<best[0]:best=(score,lag,hidden,layers,model,data,sc)
    pd.DataFrame(trial_rows).to_csv(C.METRICS/f'{a.cell}_trials.csv',index=False)
    _,lag,hidden,layers,model,data,sc=best
    Xt,yt,idx=data['test']; preds=forecast(model,Xt,sc)
    record(a.cell,y.index[idx],y.to_numpy()[idx],preds)
    cfg={'cell':a.cell,'lag':lag,'hidden':hidden,'layers':layers}
    json.dump(cfg,open(C.METRICS/f'{a.cell}_best.json','w'),indent=2)
    print('BEST:',cfg,'NOTE: use more trials and several seeds for real analysis.')
if __name__=='__main__':main()
