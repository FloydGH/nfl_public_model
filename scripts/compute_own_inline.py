import pandas as pd, numpy as np, json, unicodedata, os
from pathlib import Path
DATA_OUT=Path('data/out')

def norm_name(s):
    if not isinstance(s,str): s=str(s)
    import unicodedata
    return unicodedata.normalize('NFKD',s).strip()

def parse_gameinfo(s):
    if not isinstance(s,str) or "@" not in s: return None,None
    tok=s.split(); teams=tok[0] if tok else s
    if "@" not in teams: return None,None
    away,home=teams.split("@",1)
    return away.strip().upper(), home.strip().upper()

def safe_z_series(x):
    v=pd.to_numeric(x,errors='coerce').astype(float).values
    m=np.nanmean(v)
    sd=np.nanstd(v)
    if not np.isfinite(sd) or sd<=0: return pd.Series(np.zeros_like(v),index=x.index)
    z=(v-m)/sd
    z[~np.isfinite(z)]=0
    return pd.Series(z,index=x.index)

dk=pd.read_csv(DATA_OUT/'dk_with_proj.csv')
dk.columns=[c.strip() for c in dk.columns]
if 'team' not in dk.columns and 'TeamAbbrev' in dk.columns: dk=dk.rename(columns={'TeamAbbrev':'team'})
if 'GameInfo' not in dk.columns and 'Game Info' in dk.columns: dk=dk.rename(columns={'Game Info':'GameInfo'})
for c,d in [('proj_mean',0.0),('proj_p90',0.0),('salary',0.0),('AvgPointsPerGame',0.0),('itt',0.0)]:
    if c not in dk.columns: dk[c]=d
dk=dk[dk['salary']>0].copy()
dk['player_name']=dk['player_name'].map(norm_name)
dk['position']=dk['position'].astype(str).str.upper().str.strip()
dk['team']=dk['team'].astype(str).str.upper().str.strip()
dk['salary']=pd.to_numeric(dk['salary'],errors='coerce').fillna(0).astype(int)

if 'GameInfo' in dk.columns:
    ah=dk['GameInfo'].apply(parse_gameinfo)
    dk['away']=ah.apply(lambda x: x[0] if x else None)
    dk['home']=ah.apply(lambda x: x[1] if x else None)
    dk['opp']=np.where(dk['team'].eq(dk['home']),dk['away'],np.where(dk['team'].eq(dk['away']),dk['home'],None))
else:
    dk['opp']=None

dk['value']=dk['proj_mean']/(dk['salary']+1e-9)

temps={'QB':0.60,'RB':0.50,'WR':0.55,'TE':0.60,'DST':0.80}
dk['_z_pm']=dk.groupby('position')['proj_mean'].transform(safe_z_series)
dk['_z_val']=dk.groupby('position')['value'].transform(safe_z_series)
dk['_z_p90']=dk.groupby('position')['proj_p90'].transform(safe_z_series)
dk['_z_itt']=dk.groupby('position')['itt'].transform(safe_z_series)
eps=1e-3*(dk['salary'].rank(pct=True)+dk['AvgPointsPerGame'].rank(pct=True))
dk['score']=0.70*dk['_z_pm']+0.45*dk['_z_val']+0.20*dk['_z_p90']+0.10*dk['_z_itt']+eps

# light stack lift
qb=dk[dk.position=='QB'].copy()
team_mass={}
if len(qb)>0:
    qb['_w']=np.exp((0.6*safe_z_series(qb['proj_mean'])+0.4*safe_z_series(qb['value']))/max(temps['QB'],1e-6))
    tm=qb.groupby('team')['_w'].sum()
    mx=float(tm.max()) if len(tm)>0 else 1.0
    if mx<=0: mx=1.0
    team_mass=(tm/mx).to_dict()

lift=np.ones(len(dk))
same=(dk['position'].isin(['WR','TE'])) & dk['team'].isin(team_mass.keys())
opp =(dk['position'].isin(['WR','TE','RB'])) & dk['opp'].isin(team_mass.keys())
lift=np.where(same,lift*(1.0+0.12*dk['team'].map(team_mass).fillna(0).values),lift)
lift=np.where(opp, lift*(1.0+0.06*dk['opp'].map(team_mass).fillna(0).values),lift)

w=np.exp(dk['score'].values/np.vectorize(lambda p: temps.get(p,0.6))(dk['position'].values))*lift
w=w+1e-12

slots={'QB':1.0,'RB':2.4,'WR':3.6,'TE':1.2,'DST':1.0}
caps ={'QB':0.25,'RB':0.35,'WR':0.30,'TE':0.25,'DST':0.25}

own=np.zeros(len(dk))
for pos in slots:
    idx=np.where(dk['position'].values==pos)[0]
    if idx.size==0: continue
    target=slots[pos]; n=idx.size
    cap_eff=min(1.0, max(caps[pos], target/n + 1e-9))
    wpos=w[idx]; y=(wpos/wpos.sum())*target
    lo,hi = y.max()-cap_eff, y.max()
    for _ in range(80):
        mu=0.5*(lo+hi)
        x=np.clip(y-mu,0.0,cap_eff); s=x.sum()
        if abs(s-target)<1e-9: break
        if s>target: lo=mu
        else: hi=mu
    else:
        x=np.clip(y-mu,0.0,cap_eff); s=x.sum(); x = x if s==0 else x*(target/s)
    own[idx]=x

out=dk.copy()
out['ownership']=own.clip(0,1)
out.to_csv(DATA_OUT/'dk_with_proj_own_debug.csv',index=False)

sums=out.groupby('position')['ownership'].sum().to_dict()
print('debug CSV sums:', {k:round(v,3) for k,v in sums.items()})
print('debug nonzero by pos:', out[out.ownership>0].groupby('position').size().to_dict())
