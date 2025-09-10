import pandas as pd, numpy as np, json, unicodedata
from ..config import DATA_OUT

def parse_gameinfo(s):
    if not isinstance(s,str) or "@" not in s: return None,None
    tok=s.split(); teams=tok[0] if tok else s
    if "@" not in teams: return None,None
    away,home=teams.split("@",1)
    return away.strip().upper(), home.strip().upper()

def pct_rank(x):
    if len(x)==0: return pd.Series([],dtype=float)
    return x.rank(method='average',pct=True)

def safe_z(x):
    v=pd.to_numeric(x,errors='coerce').astype(float).values
    m=np.nanmean(v)
    sd=np.nanstd(v)
    if not np.isfinite(sd) or sd<=0:
        return pd.Series(np.zeros_like(v),index=x.index)
    z=(v-m)/sd
    z[~np.isfinite(z)]=0
    return pd.Series(z,index=x.index)

def norm_name(s):
    if not isinstance(s,str): s=str(s)
    return unicodedata.normalize('NFKD',s).strip()

def project_capped_simplex(y, cap, target, iters=100):
    low = float(y.min()) - cap
    high = float(y.max())
    for _ in range(iters):
        tau = 0.5*(low+high)
        x = np.clip(y - tau, 0.0, cap)
        s = x.sum()
        if abs(s - target) < 1e-9:
            break
        if s > target:
            low = tau
        else:
            high = tau
    x = np.clip(y - tau, 0.0, cap)
    s = x.sum()
    if s > 0 and abs(s - target) > 1e-6:
        x *= (target / s)
        x = np.clip(x, 0.0, cap)
    return x

def run():
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
    dk['_z_pm']=dk.groupby('position')['proj_mean'].transform(safe_z)
    dk['_z_val']=dk.groupby('position')['value'].transform(safe_z)
    dk['_z_p90']=dk.groupby('position')['proj_p90'].transform(safe_z)
    dk['_z_itt']=dk.groupby('position')['itt'].transform(safe_z)
    eps=1e-3*(pct_rank(dk['salary'])+pct_rank(dk['AvgPointsPerGame']))
    dk['score']=0.70*dk['_z_pm']+0.45*dk['_z_val']+0.20*dk['_z_p90']+0.10*dk['_z_itt']+eps

    qb=dk[dk.position=='QB'].copy()
    team_mass={}
    if len(qb)>0:
        qb['_w']=np.exp((0.6*safe_z(qb['proj_mean'])+0.4*safe_z(qb['value']))/max(temps['QB'],1e-6))
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
    meta={'slots':slots,'caps':caps}
    for pos in ['QB','RB','WR','TE','DST']:
        idx=np.where(dk['position'].values==pos)[0]
        if idx.size==0:
            meta[f'sum_{pos}']=0.0; meta[f'n_{pos}']=0; meta[f'cap_eff_{pos}']=0.0
            continue
        target=slots[pos]; n=idx.size
        cap_eff=min(1.0, max(caps[pos], target/n + 1e-9))
        wpos=w[idx]; y=(wpos/wpos.sum())*target
        x = project_capped_simplex(y, cap_eff, target)
        own[idx]=x
        meta[f'sum_{pos}']=float(x.sum()); meta[f'n_{pos}']=int(n); meta[f'cap_eff_{pos}']=float(cap_eff)

    out=dk.copy()
    out['ownership']=own
    out.to_csv(DATA_OUT/'dk_with_proj_own.csv',index=False)
    meta['kept_counts'] = dk['position'].value_counts().to_dict()
    try:
        with open(DATA_OUT/'ownership_meta.json','w') as f: json.dump(meta,f,indent=2)
    except Exception:
        pass

if __name__=='__main__':
    run()
