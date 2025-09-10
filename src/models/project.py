import pandas as pd, numpy as np
from ..config import DATA_INTERIM, DATA_OUT, DATA_EXTERNAL

def run():
    pb=pd.read_parquet(DATA_INTERIM/'player_baselines.parquet')
    tb=pd.read_parquet(DATA_INTERIM/'team_baselines.parquet')
    tp=pd.read_csv(DATA_INTERIM/'team_priors.csv') if (DATA_INTERIM/'team_priors.csv').exists() else pd.DataFrame(columns=['team','opp','itt'])
    m=pb.merge(tb,how='left',on='recent_team')
    if not tp.empty:
        m=m.merge(tp,how='left',left_on='recent_team',right_on='team').drop(columns=['team'])
        scale=np.where(m['team_pts_pg']>0,(m['itt']/m['team_pts_pg']).clip(0.6,1.6),1.0)
    else:
        m['itt']=np.nan
        scale=np.ones(len(m))
    m['proj_mean']=m['dk_mean']*scale
    m['proj_p90']=m['dk_p90']*scale
    proj=m[['player_id','player_name','position','recent_team','proj_mean','proj_p90','itt']].rename(columns={'recent_team':'team'})
    DATA_OUT.mkdir(parents=True,exist_ok=True)
    proj.to_csv(DATA_OUT/'projections.csv',index=False)
    if (DATA_EXTERNAL/'DKSalaries.csv').exists():
        dk=pd.read_csv(DATA_EXTERNAL/'DKSalaries.csv')
        dk.columns=[c.strip() for c in dk.columns]
        dk=dk.rename(columns={'Name':'player_name','Position':'position','TeamAbbrev':'team','Salary':'salary','Game Info':'GameInfo'})
        merged=dk.merge(proj,how='left',on=['player_name','position'])
        if 'team' not in merged.columns:
            if 'team_x' in merged.columns:
                merged['team']=merged['team_x']
            elif 'TeamAbbrev' in merged.columns:
                merged['team']=merged['TeamAbbrev']
            else:
                merged['team']=''
        for c in ['team_x','team_y','TeamAbbrev']:
            if c in merged.columns:
                merged=merged.drop(columns=[c])
        if 'AvgPointsPerGame' in merged.columns:
            mask=merged['proj_mean'].isna()
            merged.loc[mask,'proj_mean']=merged.loc[mask,'AvgPointsPerGame'].fillna(0)
            merged.loc[mask,'proj_p90']=merged.loc[mask,'proj_mean']*1.6
        dst_mask=(merged['position']=='DST') & merged['proj_mean'].isna()
        merged.loc[dst_mask,'proj_mean']=6.0
        merged.loc[dst_mask,'proj_p90']=12.0
        merged.to_csv(DATA_OUT/'dk_with_proj.csv',index=False)

if __name__=='__main__':
    run()
