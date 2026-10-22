from __future__ import annotations
import argparse, hashlib, json, time
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import root_mean_squared_error
from inf8239_u03.config import MOVIELENS_DIR, ROOT
from inf8239_u03.data import load_movielens
from inf8239_u03.metrics import catalog_coverage, hit_rate_at_k
from inf8239_u03.recommenders import MatrixFactorization, temporal_leave_one_out, weighted_popularity
p=argparse.ArgumentParser(); p.add_argument('--factors',type=int,default=20); p.add_argument('--epochs',type=int,default=12); p.add_argument('--seeds',type=int,nargs='+',default=[17,42,73]); p.add_argument('--alphas',type=float,nargs='+',default=[.25,.75]); a=p.parse_args()
ratings,movies=load_movielens(MOVIELENS_DIR); train,test=temporal_leave_one_out(ratings)
eligible=test[test.userId.isin(train.userId.unique())&test.movieId.isin(train.movieId.unique())].copy().reset_index(drop=True)
items=movies.reset_index(drop=True).copy(); X=TfidfVectorizer().fit_transform(items.genres.str.replace('|',' ',regex=False)).toarray(); ids=items.movieId.astype(int).to_numpy(); ix={int(m):i for i,m in enumerate(ids)}
train_items=set(train.movieId.astype(int)); valid_items=np.array([m for m in ids if int(m) in train_items]); pop=weighted_popularity(train,movies,quantile=0).reset_index().sort_values(['weighted_score','count'],ascending=False).movieId.astype(int).tolist()
uids=eligible.userId.astype(int).to_numpy(); userix={u:i for i,u in enumerate(uids)}; profiles=np.zeros((len(uids),X.shape[1])); seen=[set() for _ in uids]
for u,g in train[train.userId.isin(uids)].groupby('userId'):
 j=userix[int(u)]; seen[j]=set(g.movieId.astype(int)); g=g[g.movieId.isin(ix)]; ii=np.array([ix[int(m)] for m in g.movieId]); w=np.clip(g.rating.to_numpy()-2.5,.1,None); profiles[j]=w@X[ii]/w.sum()
content_scores=profiles@X.T
base_recs={'popularity':{},'content':{}}
for j,u in enumerate(uids):
 allowed=np.array([int(m) not in seen[j] for m in ids]); ci=np.flatnonzero(allowed); topc=ci[np.argsort(content_scores[j,ci])[-10:][::-1]]
 base_recs['content'][int(u)]=ids[topc].astype(int).tolist(); base_recs['popularity'][int(u)]=[m for m in pop if m not in seen[j]][:10]
rows=[]; model_rows=[]; example_rows=[]
for seed in a.seeds:
 t0=time.perf_counter(); model=MatrixFactorization(factors=a.factors,seed=seed).fit(train,epochs=a.epochs); fit_s=time.perf_counter()-t0
 uidx=np.array([model.user_index[int(u)] for u in uids]); ii=np.array([model.item_index.get(int(m),0) for m in ids]); raw=model.global_mean+model.user_factors[uidx]@model.item_factors[ii].T
 known=np.array([int(m) in model.item_index for m in ids]); raw[:,~known]=-np.inf
 for j in range(len(uids)):
  for m in seen[j]: raw[j,ix[m]]=-np.inf
 held_user=np.array([userix[int(u)] for u in eligible.userId]); held_item=np.array([ix[int(m)] for m in eligible.movieId]); pred=raw[held_user,held_item]; rmse=float(root_mean_squared_error(eligible.rating,pred))
 recs={'collaborative':{},**{f'hybrid_alpha_{alpha:g}':{} for alpha in a.alphas}}
 for j,u in enumerate(uids):
  cand=np.argpartition(raw[j],-min(100,known.sum()))[-min(100,known.sum()):]; cand=cand[np.argsort(raw[j,cand])[::-1]]
  cs=raw[j,cand]; cn=(cs-cs.min())/max(cs.max()-cs.min(),1e-12); cts=content_scores[j,cand]; ctn=(cts-cts.min())/max(cts.max()-cts.min(),1e-12)
  recs['collaborative'][int(u)]=ids[cand[:10]].astype(int).tolist()
  for alpha in a.alphas:
   order=np.argsort(alpha*cn+(1-alpha)*ctn)[::-1][:10]; recs[f'hybrid_alpha_{alpha:g}'][int(u)]=ids[cand[order]].astype(int).tolist()
 for method,r in base_recs.items(): rows.append({'seed':seed,'method':method,'rmse':float('nan'),'hit_rate_at_10':hit_rate_at_k(r,eligible,10),'catalog_coverage':catalog_coverage(r,len(movies)),'fit_seconds':0.,'model_parameters':0,'evaluated_users':len(uids),'factors':0,'epochs':0,'alpha':float('nan')})
 for method,r in recs.items():
  rows.append({'seed':seed,'method':method,'rmse':rmse,'hit_rate_at_10':hit_rate_at_k(r,eligible,10),'catalog_coverage':catalog_coverage(r,len(movies)),'fit_seconds':fit_s,'model_parameters':int(model.user_factors.size+model.item_factors.size),'evaluated_users':len(uids),'factors':a.factors,'epochs':a.epochs,'alpha':1.0 if method=='collaborative' else float(method.rsplit('_',1)[1])})
  if seed==a.seeds[0] and method=='hybrid_alpha_0.75':
   for u in list(uids)[:5]:
    for rank,mid in enumerate(r[int(u)]): example_rows.append({'userId':int(u),'movieId':int(mid),'rank':rank+1,'title':movies.set_index('movieId').loc[mid,'title']})
 print('seed',seed,'fit seconds',round(fit_s,2),flush=True)
reports=ROOT/'reports'; reports.mkdir(exist_ok=True); result=pd.DataFrame(rows); result.to_csv(reports/'seed_metrics.csv',index=False); pd.DataFrame(example_rows).to_csv(reports/'hybrid_examples.csv',index=False)
summary=result.groupby('method',dropna=False).agg(rmse=('rmse','mean'),hit_rate_at_10=('hit_rate_at_10','mean'),hit_rate_sd=('hit_rate_at_10','std'),catalog_coverage=('catalog_coverage','mean'),fit_seconds=('fit_seconds','mean'),model_parameters=('model_parameters','first'),evaluated_users=('evaluated_users','first')).reset_index(); summary.to_csv(reports/'main_results.csv',index=False)
h=result[result.method.str.startswith('hybrid_alpha_')].groupby(['factors','epochs','alpha']).agg(rmse=('rmse','mean'),rmse_sd=('rmse','std'),hit_rate_at_10=('hit_rate_at_10','mean'),hit_rate_sd=('hit_rate_at_10','std'),catalog_coverage=('catalog_coverage','mean'),fit_seconds=('fit_seconds','mean'),model_parameters=('model_parameters','first'),evaluated_users=('evaluated_users','first')).reset_index()
def dominated(i):
 r=h.iloc[i]
 return any(j!=i and o.hit_rate_at_10>=r.hit_rate_at_10 and o.catalog_coverage>=r.catalog_coverage and o.fit_seconds<=r.fit_seconds and o.model_parameters<=r.model_parameters and (o.hit_rate_at_10>r.hit_rate_at_10 or o.catalog_coverage>r.catalog_coverage or o.fit_seconds<r.fit_seconds or o.model_parameters<r.model_parameters) for j,o in h.iterrows())
h['pareto_efficient']=[not dominated(i) for i in range(len(h))]; h.to_csv(reports/'pareto_comparison.csv',index=False); h.to_json(reports/'hybrid_metrics.json',orient='records',indent=2)
pop_ids=pop[:10]; movies[movies.movieId.isin(pop_ids)].set_index('movieId').loc[pop_ids].reset_index().to_csv(reports/'cold_start_fallback.csv',index=False)
manifest={'dataset':'MovieLens Latest Small','source_url':'https://files.grouplens.org/datasets/movielens/ml-latest-small.zip','ratings':len(ratings),'movies':len(movies),'users':int(ratings.userId.nunique()),'timestamp_min':int(ratings.timestamp.min()),'timestamp_max':int(ratings.timestamp.max()),'density':len(ratings)/(ratings.userId.nunique()*movies.movieId.nunique()),'sha256':hashlib.sha256((ROOT/'data/raw/ml-latest-small.zip').read_bytes()).hexdigest(),'downloaded_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'eligible_test_events':len(eligible),'split':'latest timestamp event per user; same-time ties ordered by movieId; only train-known users/items evaluated'}
(reports/'data_audit.json').write_text(json.dumps(manifest,indent=2),encoding='utf8'); print(summary.to_string(index=False)); print(h.to_string(index=False))
