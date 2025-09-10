import requests, pandas as pd
from ..config import DATA_INTERIM, ODDS_API_KEY

def implied_totals(total,spread):
    home_itt=total/2-spread/2
    away_itt=total-home_itt
    return home_itt,away_itt

def run():
    DATA_INTERIM.mkdir(parents=True,exist_ok=True)
    if not ODDS_API_KEY:
        pd.DataFrame(columns=['game_id','home_team','away_team','total','spread','home_itt','away_itt']).to_csv(DATA_INTERIM/'odds.csv',index=False)
        return
    url='https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds?regions=us&markets=spreads,totals&oddsFormat=american&apiKey='+ODDS_API_KEY
    try:
        r=requests.get(url,timeout=20)
        r.raise_for_status()
    except Exception:
        pd.DataFrame(columns=['game_id','home_team','away_team','total','spread','home_itt','away_itt']).to_csv(DATA_INTERIM/'odds.csv',index=False)
        return
    rows=[]
    for g in r.json():
        gid=g.get('id','')
        home=g.get('home_team','')
        away=g.get('away_team','')
        total=None
        spread=None
        for b in g.get('bookmakers',[]):
            for m in b.get('markets',[]):
                if m.get('key')=='totals' and m.get('outcomes'):
                    try:
                        total=float(m['outcomes'][0]['point'])
                    except Exception:
                        pass
                if m.get('key')=='spreads' and m.get('outcomes'):
                    for o in m['outcomes']:
                        if o.get('name')==home:
                            try:
                                spread=float(o['point'])
                            except Exception:
                                pass
        if total is not None and spread is not None:
            hi,ai=implied_totals(total,spread)
            rows.append({'game_id':gid,'home_team':home,'away_team':away,'total':total,'spread':spread,'home_itt':hi,'away_itt':ai})
    pd.DataFrame(rows).to_csv(DATA_INTERIM/'odds.csv',index=False)

if __name__=='__main__':
    run()
