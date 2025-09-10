import pandas as pd, numpy as np
from ..config import DATA_OUT, DATA_EXTERNAL

def run():
    path=DATA_OUT/'dk_with_proj.csv'
    df=pd.read_csv(path)
    df.columns=[c.strip() for c in df.columns]
    if 'team' not in df.columns:
        if 'team_x' in df.columns:
            df=df.rename(columns={'team_x':'team'})
        elif 'TeamAbbrev' in df.columns:
            df=df.rename(columns={'TeamAbbrev':'team'})
    wpath=DATA_EXTERNAL/'weather.csv'
    if not wpath.exists():
        teams=df['team'].dropna().unique() if 'team' in df.columns else []
        tpl=pd.DataFrame({'team':teams,'wind_mph':0,'precip':0,'indoor':0})
        tpl.to_csv(wpath,index=False)
        df.to_csv(path,index=False)
        return
    w=pd.read_csv(wpath)
    w=w[['team','wind_mph','precip','indoor']]
    m=df.merge(w,on='team',how='left') if 'team' in df.columns else df.copy()
    for col,default in [('wind_mph',0),('precip',0),('indoor',0)]:
        if col not in m.columns: m[col]=default
        m[col]=m[col].fillna(default)
    adj=np.ones(len(m))
    wind_pen=((m['wind_mph']//5).clip(lower=0))*0.01
    wind_pen=wind_pen.clip(upper=0.08)
    mask_pass=('position' in m.columns) & m['position'].isin(['QB','WR','TE']) & (m['indoor']==0)
    adj=np.where(mask_pass,1.0-wind_pen,1.0)
    adj=np.where(mask_pass & (m['precip']>0),adj*0.95,adj)
    if 'proj_mean' in m.columns: m['proj_mean']=m['proj_mean']*adj
    if 'proj_p90' in m.columns: m['proj_p90']=m['proj_p90']*adj
    m.to_csv(path,index=False)

if __name__=='__main__':
    run()
