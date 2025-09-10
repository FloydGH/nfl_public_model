import pandas as pd
from ..config import DATA_OUT

def run():
    df=pd.read_csv(DATA_OUT/'dk_with_proj_own.csv')
    cols={c:c for c in df.columns}

    if 'player_name' not in df.columns and 'Name' in df.columns:
        df=df.rename(columns={'Name':'player_name'})
    if 'position' not in df.columns and 'Position' in df.columns:
        df=df.rename(columns={'Position':'position'})
    if 'Salary' not in df.columns and 'salary' in df.columns:
        df=df.rename(columns={'salary':'Salary'})
    if 'team' not in df.columns and 'TeamAbbrev' in df.columns:
        df=df.rename(columns={'TeamAbbrev':'team'})
    if 'GameInfo' not in df.columns and 'Game Info' in df.columns:
        df=df.rename(columns={'Game Info':'GameInfo'})

    for need, fallback, default in [
        ('player_name', None, None),
        ('position', None, None),
        ('Salary', 'salary', 0),
        ('team', 'TeamAbbrev', ''),
        ('GameInfo', None, ''),
        ('AvgPointsPerGame', None, 0.0),
        ('proj_mean', None, 0.0),
        ('proj_p90', None, 0.0),
        ('ownership', None, 0.0),
    ]:
        if need not in df.columns:
            if fallback and fallback in df.columns:
                df[need]=df[fallback]
            else:
                df[need]=default

    out=df[['player_name','team','position','Salary','GameInfo','AvgPointsPerGame','proj_mean','proj_p90','ownership']].copy()
    out.to_csv(DATA_OUT/'optimizer_input.csv',index=False)

if __name__=='__main__':
    run()
