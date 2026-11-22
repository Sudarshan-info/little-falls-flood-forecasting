"""SAU feasibility demo: binary next-hour flood classification using a 2-input first-order
Sugeno-style neuro-fuzzy classifier; validate implementation and design with supervisor.
This is NOT a certified operational warning product and not a completed dissertation.
"""
import copy,json
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler
from . import config as C
from .utils import complete_window_targets, read_hourly,forecast_splits,binary_scores

class NeuroFuzzy(nn.Module):
    def __init__(self):
        super().__init__()
        self.center=nn.Parameter(torch.tensor([[-1.,0.,1.],[-1.,0.,1.]]))
        self.log_width=nn.Parameter(torch.zeros(2,3))
        self.consequents=nn.Parameter(torch.randn(9,3)*.05)
    def forward(self,x):
        width=torch.exp(self.log_width).clamp(min=.1,max=10.)
        mf=torch.exp(-.5*((x[:,:,None]-self.center)/width)**2)
        rules=(mf[:,0,:,None]*mf[:,1,None,:]).reshape(-1,9)
        norm=rules/(rules.sum(dim=1,keepdim=True)+1e-9)
        features=torch.cat([torch.ones(len(x),1),x],dim=1)
        consequent=features@self.consequents.T
        return (norm*consequent).sum(dim=1)

def main():
    stage=read_hourly().stage_ft;s=stage.to_numpy();sp=forecast_splits(stage.index)
    at=np.flatnonzero(complete_window_targets(s[:sp['test'][1]], 4))
    stage_now=s[at-1]
    X=np.column_stack((stage_now, stage_now-s[at-4]))
    Y=(s[at]>=C.MINOR_STAGE).astype(int)
    masks={k:(at>=v[0])&(at<v[1]) for k,v in sp.items()}
    for name,mask in masks.items():
        print(name,'rows',mask.sum(),'flood hours',Y[mask].sum())
    if Y[masks['train']].sum()==0 or Y[masks['test']].sum()==0:
        raise SystemExit('STOP: train or test contains no flood hours. Change study years/station, do not force metrics.')
    sc=StandardScaler().fit(X[masks['train']]);z=sc.transform(X)
    torch.manual_seed(C.SEED);net=NeuroFuzzy();opt=torch.optim.Adam(net.parameters(),lr=.01)
    Xt=torch.tensor(z,dtype=torch.float32);y=torch.tensor(Y,dtype=torch.float32)
    ti=np.where(masks['train'])[0];vi=np.where(masks['val'])[0]
    te=np.where(masks['test'])[0]
    w=min(30.,(len(ti)-Y[ti].sum())/max(Y[ti].sum(),1))
    criterion=nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float(max(w,1))))
    state=None;best=float('inf');bad=0
    for ep in range(100):
        net.train();opt.zero_grad();loss=criterion(net(Xt[ti]),y[ti]);loss.backward();opt.step()
        net.eval()
        with torch.no_grad():val=float(criterion(net(Xt[vi]),y[vi]))
        if val<best-1e-5:best=val;bad=0;state=copy.deepcopy(net.state_dict())
        else:
            bad+=1
            if bad>=10:break
    net.load_state_dict(state);net.eval()
    with torch.no_grad():prob=torch.sigmoid(net(Xt[te])).numpy()
    # 0.5 is only a starting point; calibrate an operating threshold using validation only.
    pred=(prob>=.5).astype(int);base=(stage_now[te]>=C.MINOR_STAGE).astype(int)
    scores={'neuro_fuzzy_demo':binary_scores(Y[te],pred),'persistence_threshold':binary_scores(Y[te],base),
            'note':'ANFIS-like first-order Sugeno network with end-to-end gradient training. Supervisor approval required.'}
    json.dump(scores,open(C.METRICS/'sau_binary.json','w'),indent=2)
    dates=stage.index[at[te]]
    pd.DataFrame({'time_utc':dates,'true_flood_next_hour':Y[te],
                  'pred_flood_next_hour':pred,'prob_flood':prob,'baseline_flood':base}).to_csv(C.METRICS/'sau_binary_preds.csv',index=False)
    print(json.dumps(scores,indent=2))
    print('STOP: inspect independent flood EVENTS, false alarms, and imbalance before dissertation claims.')
if __name__=='__main__':main()
