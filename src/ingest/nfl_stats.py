import pandas as pd, numpy as np
from nfl_data_py import import_weekly_data
from ..config import DATA_INTERIM

def dk_points(df):
    pyd=0.04*df.get('passing_yards',0)+4*df.get('passing_tds',0)-1*df.get('interceptions',0)
    ryd=0.1*df.get('rushing_yards',0)+6*df.get('rushing_tds',0)
    rcd=1*df.get('receptions',0)+0.1*df.get('receiving_yards',0)+6*df.get('receiving_tds',0)
    fmb=-1*df.get('fumbles_lost',df.get('fumbles',0))
    b300=np.where(df.get('passing_yards',pd.Series(0))>=300,3,0)
    b100r=np.where(df.get('rushing_yards',pd.Series(0))>=100,3,0)
    b100rc=np.where(df.get('receiving_yards',pd.Series(0))>=100,3,0)
    two=2*df.get('two_point_conversions',0)
    return pyd+ryd+rcd+fmb+b300+b100r+b100rc+two

def run(season):
    df=import_weekly_data([season])
    for c in ['passing_yards','passing_tds','interceptions','rushing_yards','rushing_tds','receptions','receiving_yards','receiving_tds','fumbles_lost','fumbles','two_point_conversions']:
        if c not in df.columns:
            df[c]=0
    df['dk']=dk_points(df)
    g=df.groupby(['player_id','player_name','position','recent_team'],as_index=False).agg(gp=('week','count'),dk_mean=('dk','mean'),dk_p90=('dk',lambda x: np.percentile(x,90)))
    t=df.groupby(['recent_team'],as_index=False).agg(team_pts_pg=('dk','sum'))
    games=df.groupby('recent_team')['week'].nunique().reset_index(name='games')
    t=t.merge(games,how='left',on='recent_team')
    t['team_pts_pg']=t['team_pts_pg']/t['games']
    DATA_INTERIM.mkdir(parents=True,exist_ok=True)
    g.to_parquet(DATA_INTERIM/'player_baselines.parquet',index=False)
    t[['recent_team','team_pts_pg']].to_parquet(DATA_INTERIM/'team_baselines.parquet',index=False)

if __name__=='__main__':
    import sys
    season=int(sys.argv[1]) if len(sys.argv)>1 else 2024
    run(season)
