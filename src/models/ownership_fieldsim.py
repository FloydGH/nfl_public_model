import pandas as pd, numpy as np
from ..config import DATA_OUT

def softmax(x,t):
    if x.size==0:
        return x
    z=(x-x.max())/t
    e=np.exp(z)
    s=e.sum()
    return e/s if s>0 else np.full_like(x,1.0/len(x))

def build_pool(df,pos,t):
    sub=df[df.position==pos].copy()
    if sub.empty:
        return None
    vals=(sub['proj_mean'].values/(sub['salary'].values+1e-9))
    p=softmax(vals,t)
    return {
        "names":sub['player_name'].to_numpy(),
        "salary":sub['salary'].to_numpy().astype(np.int64),
        "p":p
    }

def renorm(p):
    s=p.sum()
    return p/s if s>0 else np.full_like(p,1.0/len(p))

def sample_lineup(pools,t=0.10,cap=50000,min_total=49000,max_dst_tries=6):
    qb,rb,wr,te,dst=pools
    if qb is None or rb is None or wr is None or te is None:
        return None,None
    q_idx=np.random.choice(len(qb["names"]),1,p=qb["p"])[0]
    r_idx=np.random.choice(len(rb["names"]),2,replace=False,p=rb["p"])
    w_idx=np.random.choice(len(wr["names"]),3,replace=False,p=wr["p"])
    te_idx=np.random.choice(len(te["names"]),1,p=te["p"])[0]

    picked={qb["names"][q_idx], rb["names"][r_idx[0]], rb["names"][r_idx[1]],
            wr["names"][w_idx[0]], wr["names"][w_idx[1]], wr["names"][w_idx[2]],
            te["names"][te_idx]}

    base_total=int(qb["salary"][q_idx] + rb["salary"][r_idx].sum() + wr["salary"][w_idx].sum() + te["salary"][te_idx])

    flex_names=np.concatenate([rb["names"],wr["names"],te["names"]])
    flex_salary=np.concatenate([rb["salary"],wr["salary"],te["salary"]]).astype(np.int64)
    flex_p=renorm(np.concatenate([rb["p"],wr["p"],te["p"]]))
    avail_mask=~np.isin(flex_names, list(picked))
    if not avail_mask.any():
        return None,None

    if dst is not None and len(dst["names"])>0:
        for _ in range(max_dst_tries):
            d_idx=np.random.choice(len(dst["names"]),1,p=dst["p"])[0]
            d_sal=int(dst["salary"][d_idx])
            lo=min_total - base_total - d_sal
            hi=cap - base_total - d_sal
            m=(avail_mask) & (flex_salary>=lo) & (flex_salary<=hi)
            if not m.any():
                continue
            p_sub=renorm(flex_p[m])
            f_idx=np.random.choice(p_sub.size,1,p=p_sub)[0]
            flex_name=flex_names[m][f_idx]
            flex_sal=int(flex_salary[m][f_idx])
            total=base_total + d_sal + flex_sal
            if min_total<=total<=cap:
                lu=[qb["names"][q_idx],
                    rb["names"][r_idx[0]], rb["names"][r_idx[1]],
                    wr["names"][w_idx[0]], wr["names"][w_idx[1]], wr["names"][w_idx[2]],
                    te["names"][te_idx],
                    flex_name,
                    dst["names"][d_idx]]
                return lu,total
        return None,None
    else:
        lo=min_total - base_total
        hi=cap - base_total
        m=(avail_mask) & (flex_salary>=lo) & (flex_salary<=hi)
        if not m.any():
            return None,None
        p_sub=renorm(flex_p[m])
        f_idx=np.random.choice(p_sub.size,1,p=p_sub)[0]
        flex_name=flex_names[m][f_idx]
        flex_sal=int(flex_salary[m][f_idx])
        total=base_total + flex_sal
        if min_total<=total<=cap:
            lu=[qb["names"][q_idx],
                rb["names"][r_idx[0]], rb["names"][r_idx[1]],
                wr["names"][w_idx[0]], wr["names"][w_idx[1]], wr["names"][w_idx[2]],
                te["names"][te_idx],
                flex_name]
            return lu,total
        return None,None

def run(n=30000,cap=50000,t=0.10):
    dk=pd.read_csv(DATA_OUT/'dk_with_proj.csv')
    dk=dk.dropna(subset=['proj_mean','salary'])
    dk=dk[dk['salary']>0].copy()
    dk['proj_mean']=dk['proj_mean'].clip(lower=0)

    qb=build_pool(dk,'QB',t)
    rb=build_pool(dk,'RB',t)
    wr=build_pool(dk,'WR',t)
    te=build_pool(dk,'TE',t)
    dst=build_pool(dk,'DST',t) if 'DST' in dk.position.unique() else None
    pools=(qb,rb,wr,te,dst)

    counts=dk[['player_name']].drop_duplicates().copy()
    counts['own']=0

    accepted=0
    attempts=0
    goal=n
    while accepted<goal:
        lu,total=sample_lineup(pools,t,cap)
        attempts+=1
        if lu is None:
            if attempts>goal*20:
                break
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
    run(n)
