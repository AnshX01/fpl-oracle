import json, sys, time
import numpy as np
import pandas as pd
from fpl_oracle.optimise.transfers import TransferOptimizer
rng=np.random.default_rng(90321)
rows=[]
positions=['GKP']*2+['DEF']*5+['MID']*5+['FWD']*3
# A legal owned squad and a genuine market with varied costs and projections.
for i in range(60):
    pos=positions[i%15]
    xp=float(rng.uniform(1,9))
    rows.append(dict(element=i+1,position=pos,team=i//3%20+1,value=int(rng.integers(40,80)),expected_points=xp,p10=xp*.4,p90=xp*1.8,web_name=f'P{i+1}'))
base=pd.DataFrame(rows)
squad=base.iloc[:15].copy();squad['purchase_price']=squad.value
pools={g:base.assign(expected_points=base.expected_points*np.random.default_rng(g).uniform(.7,1.4,60)) for g in [18,19]}
start=time.monotonic()
result=TransferOptimizer().evaluate_joint_transfer_and_chip_plan(current_squad_df=squad,player_pool_df=pools[18],bank=20,free_transfers=2,horizon_projections=pools,current_gw=17,target_gw=18,available_chips=['wildcard','freehit','3xc','bboost'],chips_by_set={1:['wildcard','freehit','3xc','bboost']},horizon_len=2,measure_chip_values=True,risk_preference=sys.argv[2] if len(sys.argv)>2 else 'balanced')
# Runtime is diagnostic metadata, not part of the recommendation.
result.pop('chip_measurement_seconds',None)
def clean(v):
    if isinstance(v,pd.DataFrame):return clean(v.to_dict('records'))
    if isinstance(v,dict):return {str(k):clean(x) for k,x in v.items()}
    if isinstance(v,(list,tuple,set,frozenset)):return [clean(x) for x in v]
    if isinstance(v,np.generic):return v.item()
    return v
with open(sys.argv[1],'w') as f:json.dump(clean(result),f,sort_keys=True,default=str)
print(sys.argv[1],round(time.monotonic()-start,2),flush=True)
