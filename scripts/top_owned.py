import pandas as pd, sys
p='data/out/dk_with_proj_own.csv' if len(sys.argv)<2 else sys.argv[1]
df=pd.read_csv(p)
if 'Salary' not in df.columns and 'salary' in df.columns: df=df.rename(columns={'salary':'Salary'})
keep=[c for c in ['player_name','team','position','Salary','proj_mean','proj_p90','ownership'] if c in df.columns]
df=df[keep].sort_values('ownership',ascending=False)
print(df.head(10).to_string(index=False))
for pos in ['QB','RB','WR','TE','DST']:
    d=df[df['position']==pos].head(10)
    if len(d)>0:
        print('\n'+pos)
        print(d[['player_name','team','Salary','proj_mean','ownership']].to_string(index=False))
