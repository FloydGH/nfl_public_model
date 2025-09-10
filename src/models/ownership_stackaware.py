import pandas as pd, numpy as np
from ..config import DATA_OUT

def softmax(x,t):
    if x.size==0: return x
    z=(x-x.max())/t
    e=np.exp(z)
    s=e.sum()
    return e/s if s>0 else np.full_like(x,1.0/len(x))

def parse_gameinfo(s):
    if not isinstance(s,str) or "@" not in s: return None,None
    tok=s.split()
    teams=tok[0] if tok else s
    if "@" not in teams: return None,None
    parts=teams.split("@")
    away=parts[0].strip().upper()
    home=parts[1].strip().upper()
    return away,home

def build_indices(df):
    teams=df['team'].dropna().unique().tolist()
    qbs={}; passcatch={}; rbs={}; defs={}
    for tm in teams:
        qbs[tm]=df[(df.team==tm)&(df.position=='QB')].index.values
        passcatch[tm]=df[(df.team==tm)&(df.position.isin(['WR','TE']))].index.values
        rbs[tm]=df[(df.team==tm)&(df.position=='RB')].index.values
        defs[tm]=df[(df.team==tm)&(df.position=='DST')].index.values
    return qbs,passcatch,rbs,defs

def weighted_idx(idx,vals,t):
    if idx.size==0: return None
    w=softmax(vals[idx],t)
    i=np.random.choice(idx.size,1,p=w)[0]
    return idx[i]

def sample_lineup(df,qbs,pc,rbs,defs,bring_back_p=0.6,t=0.10,min_total=49000,cap=50000,max_tries=12):
    qb_pool=np.concatenate([v for v in qbs.values() if v.size>0]) if qbs else np.array([],dtype=int)
    if qb_pool.size==0: return None,None
    qb_idx=weighted_idx(qb_pool,df['value'].values,t)
    if qb_idx is None: return None,None
    qb_team=df.at[qb_idx,'team']
    opp_team=df.at[qb_idx,'opp']
    stack_k=np.random.choice([1,2,3,0],p=[0.25,0.6,0.1,0.05])
    chosen=set([qb_idx])
    total=int(df.at[qb_idx,'salary'])
    need_rb=2; need_wr=3; need_te=1; need_dst=1 if df.position.eq('DST').any() else 0
    pool_pc=pc.get(qb_team,np.array([],dtype=int))
    if pool_pc.size<stack_k: stack_k=pool_pc.size
    for _ in range(stack_k):
        i=weighted_idx(pool_pc,df['value'].values,t)
        if i is None or i in chosen: continue
        chosen.add(i); total+=int(df.at[i,'salary'])
        pos=df.at[i,'position']
        if pos=='WR': need_wr=max(0,need_wr-1)
        elif pos=='TE': need_te=max(0,need_te-1)
        elif pos=='RB': need_rb=max(0,need_rb-1)
    if opp_team and np.random.rand()<bring_back_p:
        opp_pc=np.concatenate([pc.get(opp_team,np.array([],dtype=int)), rbs.get(opp_team,np.array([],dtype=int))])
        opp_pc=opp_pc[~np.isin(opp_pc,list(chosen))]
        if opp_pc.size>0:
            j=weighted_idx(opp_pc,df['value'].values,t)
            if j is not None:
                chosen.add(j); total+=int(df.at[j,'salary'])
                pos=df.at[j,'position']
                if pos=='WR': need_wr=max(0,need_wr-1)
                elif pos=='TE': need_te=max(0,need_te-1)
                elif pos=='RB': need_rb=max(0,need_rb-1)
    for _ in range(max_tries):
        sel=list(chosen)
        all_rb=np.concatenate([v for v in rbs.values() if v.size>0]) if rbs else np.array([],dtype=int)
        all_wr=np.concatenate([v for v in pc.values() if v.size>0]) if pc else np.array([],dtype=int)
        all_te=df[df.position=='TE'].index.values
        all_dst=np.concatenate([v for v in defs.values() if v.size>0]) if defs else np.array([],dtype=int)
        pool_rb=all_rb[~np.isin(all_rb,sel)]
        pool_wr=all_wr[~np.isin(all_wr,sel)]
        pool_te=all_te[~np.isin(all_te,sel)]
        pool_dst=all_dst[~np.isin(all_dst,sel)]
        cur_total=total
        rb_pick=[]
        for _r in range(need_rb):
            if pool_rb.size==0: break
            r=weighted_idx(pool_rb,df['value'].values,t)
            if r is None: break
            rb_pick.append(r); cur_total+=int(df.at[r,'salary'])
            pool_rb=pool_rb[pool_rb!=r]
        if len(rb_pick)<need_rb: continue
        wr_pick=[]
        for _w in range(need_wr):
            if pool_wr.size==0: break
            w=weighted_idx(pool_wr,df['value'].values,t)
            if w is None: break
            wr_pick.append(w); cur_total+=int(df.at[w,'salary'])
            pool_wr=pool_wr[pool_wr!=w]
        if len(wr_pick)<need_wr: continue
        te_pick=[]
        for _t in range(need_te):
            if pool_te.size==0: break
            tei=weighted_idx(pool_te,df['value'].values,t)
            if tei is None: break
            te_pick.append(tei); cur_total+=int(df.at[tei,'salary'])
            pool_te=pool_te[pool_te!=tei]
        if len(te_pick)<need_te: continue
        flex_pool=np.concatenate([pool_rb,pool_wr,pool_te])
        flex_pool=flex_pool[~np.isin(flex_pool,rb_pick+wr_pick+te_pick)]
        if flex_pool.size==0: continue
        remaining_lo=49000-cur_total-(df.loc[pool_dst,'salary'].min() if need_dst and pool_dst.size>0 else 0)
        remaining_hi=50000-cur_total-(df.loc[pool_dst,'salary'].max() if need_dst and pool_dst.size>0 else 0)
        flex_mask=(df.loc[flex_pool,'salary'].values>=remaining_lo)&(df.loc[flex_pool,'salary'].values<=remaining_hi)
        if not flex_mask.any(): continue
        flex_w=softmax(df.loc[flex_pool,'value'].values,t)
        flex_w=flex_w*flex_mask
        s=flex_w.sum()
        if s==0: continue
        f=np.random.choice(flex_pool,1,p=flex_w/s)[0]
        cur_total+=int(df.at[f,'salary'])
        d_pick=[]
        if need_dst and pool_dst.size>0:
            dst_lo=49000-cur_total
            dst_hi=50000-cur_total
            dst_sal=df.loc[pool_dst,'salary'].values
            ok=(dst_sal>=dst_lo)&(dst_sal<=dst_hi)
            if ok.any():
                w=softmax(df.loc[pool_dst,'value'].values,t)
                w=w*ok; s=w.sum()
                if s>0:
                    d=np.random.choice(pool_dst,1,p=w/s)[0]
                    d_pick.append(d); cur_total+=int(df.at[d,'salary'])
        final_sel=sel+rb_pick+wr_pick+te_pick+[f]+d_pick
        if 49000<=cur_total<=50000 and len(final_sel)>=8:
            names=df.loc[final_sel,'player_name'].tolist()
            return names,cur_total
    return None,None

def run(n=30000,t=0.10):
    dk=pd.read_csv(DATA_OUT/'dk_with_proj.csv')
    dk.columns=[c.strip() for c in dk.columns]
    if 'team' not in dk.columns:
        if 'team_x' in dk.columns: dk=dk.rename(columns={'team_x':'team'})
        elif 'TeamAbbrev' in dk.columns: dk=dk.rename(columns={'TeamAbbrev':'team'})
        else: dk['team']=''
    if 'GameInfo' not in dk.columns and 'Game Info' in dk.columns:
        dk=dk.rename(columns={'Game Info':'GameInfo'})
    dk=dk.dropna(subset=['proj_mean','salary'])
    dk=dk[dk['salary']>0].copy()
    dk['proj_mean']=dk['proj_mean'].clip(lower=0)
    dk['value']=dk['proj_mean']/(dk['salary']+1e-9)
    away_home=dk['GameInfo'].apply(parse_gameinfo) if 'GameInfo' in dk.columns else None
    if away_home is not None:
        dk['away']=away_home.apply(lambda x: x[0] if x else None)
        dk['home']=away_home.apply(lambda x: x[1] if x else None)
        dk['opp']=np.where(dk['team'].eq(dk['home']),dk['away'],np.where(dk['team'].eq(dk['away']),dk['home'],None))
    else:
        dk['opp']=None
    qbs,pc,rbs,defs=build_indices(dk)
    counts=dk[['player_name']].drop_duplicates().copy()
    counts['own']=0
    accepted=0
    attempts=0
    goal=n
    while accepted<goal:
        lu,total=sample_lineup(dk,qbs,pc,rbs,defs,bring_back_p=0.6,t=t,min_total=49000,cap=50000,max_tries=16)
        attempts+=1
        if lu is None:
            if attempts>goal*40: break
            continue
        counts.loc[counts.player_name.isin(lu),'own']+=1
        accepted+=1
    denom=max(1,accepted)
    counts['ownership']=counts['own']/denom
    out=dk.merge(counts[['player_name','ownership']],on='player_name',how='left')
    out['ownership']=out['ownership'].fillna(0)
    out.to_csv(DATA_OUT/'dk_with_proj_own.csv',index=False)

if __name__=='__main__':
    import sys
    n=int(sys.argv[1]) if len(sys.argv)>1 else 30000
    run(n,0.10)
