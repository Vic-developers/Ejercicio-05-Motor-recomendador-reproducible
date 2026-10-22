from __future__ import annotations
import argparse, hashlib, json, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import root_mean_squared_error
from inf8239_u03.config import MOVIELENS_DIR, ROOT
from inf8239_u03.data import load_movielens
from inf8239_u03.metrics import catalog_coverage, hit_rate_at_k
from inf8239_u03.recommenders import ContentRecommender, MatrixFactorization, recommend_for_user, temporal_leave_one_out, weighted_popularity

parser=argparse.ArgumentParser(description='Compara métodos, perfiles y costos con corte temporal.')
parser.add_argument('--configs', nargs=2, type=int, action='append', metavar=('FACTORS','EPOCHS'), default=None)
parser.add_argument('--seeds', type=int, nargs='+', default=[17,42,73])
parser.add_argument('--alphas', type=float, nargs='+', default=[0.25,0.75])
args=parser.parse_args()
configs=args.configs or [[8,6],[20,6],[20,12]]
ratings,movies=load_movielens(MOVIELENS_DIR); train,test=temporal_leave_one_out(ratings)
eligible=test[test.userId.isin(train.userId.unique()) & test.movieId.isin(train.movieId.unique())].copy().reset_index(drop=True)
items=movies.reset_index(drop=True).copy(); X=TfidfVectorizer().fit_transform(items.genres.str.replace('|',' ',regex=False)).toarray(); ids=items.movieId.astype(int).to_numpy(); item_row={int(m):i for i,m in enumerate(ids)}
content_model=ContentRecommender().fit(movies)
pop=weighted_popularity(train,movies).reset_index().sort_values(['weighted_score','count','movieId'],ascending=[False,False,True]).movieId.astype(int).tolist()
uids=eligible.userId.astype(int).to_numpy(); user_row={u:i for i,u in enumerate(uids)}; profiles=np.zeros((len(uids),X.shape[1])); seen=[set() for _ in uids]
for uid, history in train[train.userId.isin(uids)].groupby('userId'):
    row=user_row[int(uid)]; seen[row]=set(history.movieId.astype(int)); history=history[history.movieId.isin(item_row)]
    loc=np.array([item_row[int(m)] for m in history.movieId]); weights=np.clip(history.rating.to_numpy()-2.5,.1,None)
    profiles[row]=weights @ X[loc] / weights.sum()
content_scores=profiles @ X.T
recs_by_profile={'popularity':{},'content':{}}
for row,uid in enumerate(uids):
    allowed=np.array([int(m) not in seen[row] for m in ids]); candidates=np.flatnonzero(allowed)
    order=candidates[np.argsort(content_scores[row,candidates])[-10:][::-1]]
    recs_by_profile['content'][int(uid)]=ids[order].astype(int).tolist()
    recs_by_profile['popularity'][int(uid)]=[m for m in pop if m not in seen[row]][:10]

hist_counts=train.groupby('userId').size(); short_cut=float(hist_counts.quantile(.25)); broad_cut=float(hist_counts.quantile(.75))
profile_of={int(uid):('historial_corto' if hist_counts.loc[uid]<=short_cut else 'historial_amplio' if hist_counts.loc[uid]>=broad_cut else 'historial_intermedio') for uid in uids}
all_rows=[]; profile_rows=[]; error_rows=[]; first_model=None
for factors,epochs in configs:
    config=f'{factors}f_{epochs}e'
    for seed in args.seeds:
        start=time.perf_counter(); model=MatrixFactorization(factors=factors,seed=seed).fit(train,epochs=epochs); elapsed=time.perf_counter()-start
        if first_model is None: first_model=model
        uidx=np.array([model.user_index[int(uid)] for uid in uids]); iidx=np.array([model.item_index.get(int(mid),0) for mid in ids])
        predicted=model.global_mean+model.user_factors[uidx] @ model.item_factors[iidx].T
        known=np.array([int(mid) in model.item_index for mid in ids]); predicted[:,~known]=-np.inf
        for row in range(len(uids)):
            for mid in seen[row]: predicted[row,item_row[mid]]=-np.inf
        held_rows=np.array([user_row[int(uid)] for uid in eligible.userId]); held_items=np.array([item_row[int(mid)] for mid in eligible.movieId])
        rmse=float(root_mean_squared_error(eligible.rating,predicted[held_rows,held_items]))
        current={'collaborative':{}}
        current.update({f'hybrid_alpha_{alpha:g}':{} for alpha in args.alphas})
        for row,uid in enumerate(uids):
            cand_count=min(100,int(known.sum())); cand=np.argpartition(predicted[row],-cand_count)[-cand_count:]
            cand=cand[np.argsort(predicted[row,cand])[::-1]]
            collab=predicted[row,cand]; collab=(collab-collab.min())/max(collab.max()-collab.min(),1e-12)
            cont=content_scores[row,cand]; cont=(cont-cont.min())/max(cont.max()-cont.min(),1e-12)
            current['collaborative'][int(uid)]=ids[cand[:10]].astype(int).tolist()
            for alpha in args.alphas:
                order=np.argsort(alpha*collab+(1-alpha)*cont)[::-1][:10]
                current[f'hybrid_alpha_{alpha:g}'][int(uid)]=ids[cand[order]].astype(int).tolist()
        methods={**(recs_by_profile if config==f'{configs[0][0]}f_{configs[0][1]}e' and seed==args.seeds[0] else {}),**current}
        for method,rec_lists in methods.items():
            all_rows.append({'config':config,'seed':seed,'method':method,'factors':factors if method not in ('popularity','content') else 0,'epochs':epochs if method not in ('popularity','content') else 0,'alpha':(float(method.rsplit('_',1)[1]) if method.startswith('hybrid_alpha_') else (1.0 if method=='collaborative' else np.nan)),'rmse':(rmse if method not in ('popularity','content') else np.nan),'hit_rate_at_10':hit_rate_at_k(rec_lists,eligible,10),'catalog_coverage':catalog_coverage(rec_lists,len(movies)),'fit_seconds':(elapsed if method not in ('popularity','content') else 0.0),'model_parameters':(int(model.user_factors.size+model.item_factors.size) if method not in ('popularity','content') else 0),'evaluated_users':len(uids)})
            for segment in ('historial_corto','historial_amplio'):
                group_ids=[int(u) for u in uids if profile_of[int(u)]==segment]
                truth=eligible[eligible.userId.isin(group_ids)]
                profile_rows.append({'config':config,'seed':seed,'method':method,'profile':segment,'users':len(group_ids),'hit_rate_at_10':hit_rate_at_k({u:rec_lists[u] for u in group_ids},truth,10),'catalog_coverage':catalog_coverage({u:rec_lists[u] for u in group_ids},len(movies))})
        # Keep a concise, inspectable sample of held-out misses for main methods.
        if seed==args.seeds[0] and config==f'{configs[0][0]}f_{configs[0][1]}e':
            for method,rec_lists in methods.items():
                if method not in ('popularity','content','collaborative','hybrid_alpha_0.75'): continue
                for row in range(len(uids)):
                    uid=int(uids[row]); target=eligible.loc[eligible.userId.eq(uid)].iloc[0]; rec=rec_lists[uid]
                    error_rows.append({'config':config,'seed':seed,'userId':uid,'profile':profile_of[uid],'method':method,'heldout_movieId':int(target.movieId),'heldout_title':movies.set_index('movieId').loc[int(target.movieId),'title'],'heldout_rating':float(target.rating),'hit_at_10':int(int(target.movieId) in rec),'rank_if_hit':(rec.index(int(target.movieId))+1 if int(target.movieId) in rec else None),'top_recommendations':' | '.join(movies.set_index('movieId').loc[mid,'title'] for mid in rec[:5])})
        print(f'config={config} seed={seed} fit={elapsed:.2f}s RMSE={rmse:.4f}',flush=True)

reports=ROOT/'reports'; reports.mkdir(exist_ok=True); result=pd.DataFrame(all_rows); result.to_csv(reports/'seed_metrics.csv',index=False); pd.DataFrame(profile_rows).to_csv(reports/'profile_metrics.csv',index=False); pd.DataFrame(error_rows).to_csv(reports/'error_analysis.csv',index=False)
main=result.groupby(['config','method'],dropna=False).agg(rmse=('rmse','mean'),rmse_sd=('rmse','std'),hit_rate_at_10=('hit_rate_at_10','mean'),hit_rate_sd=('hit_rate_at_10','std'),catalog_coverage=('catalog_coverage','mean'),fit_seconds=('fit_seconds','mean'),model_parameters=('model_parameters','first'),evaluated_users=('evaluated_users','first')).reset_index(); main.to_csv(reports/'main_results.csv',index=False)
hybrid=result[result.method.str.startswith('hybrid_alpha_')].groupby(['config','factors','epochs','alpha']).agg(rmse=('rmse','mean'),rmse_sd=('rmse','std'),hit_rate_at_10=('hit_rate_at_10','mean'),hit_rate_sd=('hit_rate_at_10','std'),catalog_coverage=('catalog_coverage','mean'),fit_seconds=('fit_seconds','mean'),model_parameters=('model_parameters','first'),evaluated_users=('evaluated_users','first')).reset_index()
def dominated(i):
    r=hybrid.iloc[i]
    return any(j!=i and x.hit_rate_at_10>=r.hit_rate_at_10 and x.catalog_coverage>=r.catalog_coverage and x.fit_seconds<=r.fit_seconds and x.model_parameters<=r.model_parameters and (x.hit_rate_at_10>r.hit_rate_at_10 or x.catalog_coverage>r.catalog_coverage or x.fit_seconds<r.fit_seconds or x.model_parameters<r.model_parameters) for j,x in hybrid.iterrows())
hybrid['pareto_efficient']=[not dominated(i) for i in range(len(hybrid))]; hybrid.to_csv(reports/'pareto_comparison.csv',index=False)
hybrid.to_json(reports/'hybrid_metrics.json',orient='records',indent=2)
# Runtime proof for a no-history user and a new user with a small known-item history.
if first_model is None: first_model=MatrixFactorization(factors=configs[0][0],seed=args.seeds[0]).fit(train,epochs=configs[0][1])
cold=recommend_for_user(-1,train,movies,first_model,content_model,alpha=.75,k=10)
new_id=-2; sample_history=train.iloc[:3].copy(); sample_history['userId']=new_id
new_user=recommend_for_user(new_id,sample_history,movies,first_model,content_model,alpha=.75,k=10)
assert len(cold)==10 and cold.recommendation_source.eq('global_popularity_cold_user').all()
assert len(new_user)==10 and new_user.recommendation_source.eq('content_cold_user_with_history').all()
(reports/'cold_start_validation.json').write_text(json.dumps({'no_history_user':{'source':cold.recommendation_source.iloc[0],'recommendations':cold.title.tolist(),'count':len(cold)},'new_user_with_3_known_movies':{'source':new_user.recommendation_source.iloc[0],'recommendations':new_user.title.tolist(),'count':len(new_user)},'assertions':'both fallbacks returned 10 unseen recommendations'},ensure_ascii=False,indent=2),encoding='utf-8')
pop_ids=pop[:10]; movies[movies.movieId.isin(pop_ids)].set_index('movieId').loc[pop_ids].reset_index().to_csv(reports/'cold_start_fallback.csv',index=False)
manifest={'dataset':'MovieLens Latest Small','source_url':'https://files.grouplens.org/datasets/movielens/ml-latest-small.zip','ratings':len(ratings),'movies':len(movies),'users':int(ratings.userId.nunique()),'timestamp_min':int(ratings.timestamp.min()),'timestamp_max':int(ratings.timestamp.max()),'density':len(ratings)/(ratings.userId.nunique()*movies.movieId.nunique()),'sha256':hashlib.sha256((ROOT/'data/raw/ml-latest-small.zip').read_bytes()).hexdigest(),'downloaded_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'eligible_test_events':len(eligible),'excluded_test_events':int(len(test)-len(eligible)),'history_thresholds':{'short_max_train_ratings':short_cut,'broad_min_train_ratings':broad_cut},'split':'latest timestamp event per user; same-time ties ordered by movieId; only train-known users/items evaluated'}
(reports/'data_audit.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
print(main.to_string(index=False)); print(hybrid[['config','alpha','hit_rate_at_10','catalog_coverage','fit_seconds','model_parameters','pareto_efficient']].to_string(index=False))
