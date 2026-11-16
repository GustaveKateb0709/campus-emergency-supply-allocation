# -*- coding: utf-8 -*-
"""训练阶段改变人力成本权重（0.1 / 0.5），检验策略是否随权重改变"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from campus_env import CampusSupplyEnv
from evaluate import evaluate_policy, make_policy_fn
from policies import StaticPolicy, GreedyPolicy

BASE=os.path.dirname(os.path.abspath(__file__))
BC=os.path.join(BASE,"Train Models","bc_policy.zip")
OUT=os.path.join(BASE,"Train Models","weights"); os.makedirs(OUT,exist_ok=True)
TOTAL=350_000; N_ENVS=8; EVAL_EVERY=50_000; N_EVAL=60
SEEDS=[0,1,2,3,4]

class RewardScale(gym.Wrapper):
    def __init__(self,env,scale=0.02):
        super().__init__(env); self.scale=scale
    def step(self,a):
        o,r,t,tr,i=self.env.step(a); return o,r*self.scale,t,tr,i

class SeededEnv(gym.Wrapper):
    def __init__(self,env,base):
        super().__init__(env); self.base=base; self._ep=0
    def reset(self,**kw):
        kw.pop("seed",None); self._ep+=1
        return self.env.reset(seed=self.base+self._ep,**kw)

def lin(x): return lambda p: x*p

def make_env(w,seed,rank):
    def _f():
        base=3_000_000+int(round(w*1_000_000))+seed*10_000+rank*1_000
        return SeededEnv(RewardScale(CampusSupplyEnv(transport_weight=w),0.02),base)
    return _f

def train(w,seed):
    venv=VecMonitor(DummyVecEnv([make_env(w,seed,i) for i in range(N_ENVS)]))
    m=PPO("MlpPolicy",venv,seed=seed,verbose=0,learning_rate=lin(5e-4),
          n_steps=512,batch_size=512,gamma=0.99,gae_lambda=0.95,ent_coef=0.003,
          policy_kwargs=dict(net_arch=[64,64]))
    if os.path.exists(BC):
        bc=PPO.load(BC,device="cpu"); m.policy.load_state_dict(bc.policy.state_dict())
    s=0
    while s<TOTAL:
        m.learn(total_timesteps=EVAL_EVERY,reset_num_timesteps=False,progress_bar=False); s+=EVAL_EVERY
    m.save(os.path.join(OUT,f"ppo_w{w}_seed{seed}"))
    return m

res={}
for w in [0.1,0.5]:
    print(f"\n===== 训练权重 w={w} =====",flush=True)
    ek=dict(transport_weight=w)
    sh,co,se=[],[],[]
    for seed in SEEDS:
        m=train(w,seed)
        fn=lambda o,mm=m: mm.predict(o,deterministic=True)[0]
        fn.reset=lambda: None
        s,_=evaluate_policy(fn,n_episodes=N_EVAL,env_kwargs=ek)
        sh.append(s['shortage'][0]); co.append(s['cost'][0]); se.append(s['service'][0])
        print(f"  seed{seed}: 缺口 {s['shortage'][0]:.1f} 成本 {s['cost'][0]:.1f}",flush=True)
    res[str(w)]={"shortage":float(np.mean(sh)),"shortage_std":float(np.std(sh)),
                 "cost":float(np.mean(co)),"cost_std":float(np.std(co)),
                 "service":float(np.mean(se))}
    r=res[str(w)]
    print(f"  => 缺口 {r['shortage']:.1f}±{r['shortage_std']:.1f} | 成本 {r['cost']:.1f}±{r['cost_std']:.1f} | 保障率 {r['service']:.4f}",flush=True)

json.dump(res,open(os.path.join(BASE,"results_weights.json"),"w"),ensure_ascii=False,indent=2)
print("\n已保存 results_weights.json")
