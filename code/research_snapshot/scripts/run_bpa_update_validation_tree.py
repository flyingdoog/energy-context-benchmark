"""Default LightGBM chronological-validation replication with fixed-width inputs."""
from concurrent.futures import ThreadPoolExecutor,as_completed
import argparse
import json
import time
import run_local
import numpy as np
from lightgbm import Booster,LGBMRegressor
from bpa_update_validation_common import DEST,CONDITIONS,load,load_split,tabular,evaluation_parts,write_prediction


def fit(condition,packets,train,validation):
    folder=DEST/f'lightgbm_{condition}';folder.mkdir(exist_ok=True)
    receipt=folder/'receipt.json'
    if receipt.exists() and json.loads(receipt.read_text()).get('completed'):
        return json.loads(receipt.read_text())
    start=time.perf_counter();p=packets['winter'];x=tabular(p,train,condition);y=p['target'][train,2:]
    columns=x.shape[1]
    for lead in range(10):
        path=folder/f'lead_{lead:02d}.txt'
        if not path.exists():
            model=LGBMRegressor(n_jobs=2,random_state=2027,verbosity=-1)
            model.fit(x,y[:,lead]);model.booster_.save_model(str(path))
        print(json.dumps({'model':'lightgbm','condition':condition,'trained_leads':lead+1,
                          'seconds':round(time.perf_counter()-start,1)}),flush=True)
    del x
    # Predict one period at a time to bound memory when fits run concurrently.
    for period,p,indices in evaluation_parts(packets,validation):
        pred=np.empty((len(indices),10),dtype=np.float32)
        x=tabular(p,indices,condition)
        for lead in range(10):
            model=Booster(model_file=str(folder/f'lead_{lead:02d}.txt'))
            pred[:,lead]=np.maximum(model.predict(x,num_threads=2),0)
        write_prediction(folder/f'{period}.npz',p['meta'].iloc[indices],pred);del x
    result=dict(completed=True,family='lightgbm',condition=condition,features=columns,
                train_cases=len(train),seconds=time.perf_counter()-start,
                recipe='Default100trees/31leaves/MSE/seed2027. Same input width across conditions; inactive feedback columns zero. No HPO.')
    receipt.write_text(json.dumps(result,indent=2)+'\n');return result


def main():
    global DEST,tabular
    parser=argparse.ArgumentParser();parser.add_argument('--encoding',choices=['raw','typed'],default='raw')
    args=parser.parse_args()
    if args.encoding=='typed':
        from bpa_typed_inputs import tabular as typed_tabular
        tabular=typed_tabular;DEST=DEST.parent/'update_validation_typed';DEST.mkdir(exist_ok=True)
    train,validation=load_split();packets={s:load(s) for s in ['winter','spring','summer']}
    results=[];start=time.perf_counter()
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(fit,c,packets,train,validation) for c in CONDITIONS]
        for future in as_completed(futures):
            result=future.result();results.append(result)
            print(json.dumps({'completed':result['condition'],'done':len(results),'total':4}),flush=True)
    (DEST/'tree_receipt.json').write_text(json.dumps(dict(completed=True,fits=results,
        wall_seconds=time.perf_counter()-start),indent=2)+'\n')


if __name__=='__main__':main()
