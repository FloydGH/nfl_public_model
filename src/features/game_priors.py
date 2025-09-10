import pandas as pd
from ..config import DATA_INTERIM
def run():
    p=pd.read_csv(DATA_INTERIM/'odds.csv')
    if p.empty:
        pd.DataFrame(columns=['team','opp','itt']).to_csv(DATA_INTERIM/'team_priors.csv',index=False)
        return
    a=p[['away_team','home_team','away_itt']].rename(columns={'away_team':'team','home_team':'opp','away_itt':'itt'})
    b=p[['home_team','away_team','home_itt']].rename(columns={'home_team':'team','away_team':'opp','home_itt':'itt'})
    out=pd.concat([a,b],ignore_index=True)
    out.to_csv(DATA_INTERIM/'team_priors.csv',index=False)
if __name__=='__main__':
    run()
