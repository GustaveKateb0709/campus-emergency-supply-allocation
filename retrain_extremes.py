# -*- coding: utf-8 -*-
"""在需求强度两端（ds=0.5、ds=2.5）各重训 5 个 PPO 种子，检验敏感性趋势是否为分布外假象"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from campus_env import CampusSupplyEnv
from evaluate import evaluate_policy, make_policy_fn
from policies import StaticPolicy, GreedyPolicy

BASE = os.path.dirname(os.path.abspath(__file__))
BC_PATH = os.path.join(BASE, "Train Models", "bc_policy.zip")
OUT_MODEL = os.path.join(BASE, "Train Models", "extremes")
os.makedirs(OUT_MODEL, exist_ok=True)

TOTAL_STEPS = 350_000
N_ENVS = 8
EVAL_EVERY = 50_000
EVAL_EPS = 10
N_EVAL = 60
SEEDS = [0, 1, 2, 3, 4]


class RewardScale(gym.Wrapper):
    def __init__(self, env, scale=0.02):
        super().__init__(env); self.scale = scale
    def step(self, action):
        obs, r, term, trunc, info = self.env.step(action)
        return obs, r * self.scale, term, trunc, info


class SeededEnv(gym.Wrapper):
    def __init__(self, env, seed_base):
        super().__init__(env); self.seed_base = seed_base; self._ep = 0
    def reset(self, **kwargs):
        kwargs.pop("seed", None); self._ep += 1
        return self.env.reset(seed=self.seed_base + self._ep, **kwargs)


def linear_schedule(init):
    return lambda progress: init * progress


def make_env(ds, seed, rank):
    def _f():
        base = 2_000_000 + int(round(ds * 100_000)) + seed * 10_000 + rank * 1_000
        return SeededEnv(RewardScale(CampusSupplyEnv(demand_scale=ds), 0.02), base)
    return _f


def train(ds, seed):
    venv = VecMonitor(DummyVecEnv([make_env(ds, seed, i) for i in range(N_ENVS)]))
    model = PPO("MlpPolicy", venv, seed=seed, verbose=0,
                learning_rate=linear_schedule(5e-4), n_steps=512, batch_size=512,
                gamma=0.99, gae_lambda=0.95, ent_coef=0.003,
                policy_kwargs=dict(net_arch=[64, 64]))
    if os.path.exists(BC_PATH):
        bc = PPO.load(BC_PATH, device="cpu")
        model.policy.load_state_dict(bc.policy.state_dict())
    step = 0
    while step < TOTAL_STEPS:
        model.learn(total_timesteps=EVAL_EVERY, reset_num_timesteps=False, progress_bar=False)
        step += EVAL_EVERY
    model.save(os.path.join(OUT_MODEL, f"ppo_ds{ds}_seed{seed}"))
    return model


res = {}
for ds in [0.5, 2.5]:
    print(f"\n===== 需求强度 ds={ds} 重训 =====", flush=True)
    ek = dict(demand_scale=ds)
    s_static, _ = evaluate_policy(make_policy_fn(StaticPolicy()), n_episodes=N_EVAL, env_kwargs=ek)
    s_greedy, _ = evaluate_policy(make_policy_fn(GreedyPolicy()), n_episodes=N_EVAL, env_kwargs=ek)
    ppo_short, ppo_service, ppo_cost = [], [], []
    for seed in SEEDS:
        m = train(ds, seed)
        fn = lambda obs, mm=m: mm.predict(obs, deterministic=True)[0]
        fn.reset = lambda: None
        s, _ = evaluate_policy(fn, n_episodes=N_EVAL, env_kwargs=ek)
        ppo_short.append(s['shortage'][0]); ppo_service.append(s['service'][0]); ppo_cost.append(s['cost'][0])
        print(f"  seed{seed}: 缺口 {s['shortage'][0]:.1f} 保障率 {s['service'][0]:.4f}", flush=True)
    res[str(ds)] = {
        "static_shortage": s_static['shortage'][0],
        "greedy_shortage": s_greedy['shortage'][0],
        "ppo_shortage_mean": float(np.mean(ppo_short)),
        "ppo_shortage_std": float(np.std(ppo_short)),
        "ppo_service_mean": float(np.mean(ppo_service)),
        "ppo_cost_mean": float(np.mean(ppo_cost)),
        "ppo_seeds": [float(x) for x in ppo_short],
    }
    r = res[str(ds)]
    print(f"  => 静态 {r['static_shortage']:.1f} | 贪婪 {r['greedy_shortage']:.1f} | 重训PPO {r['ppo_shortage_mean']:.1f}±{r['ppo_shortage_std']:.1f}", flush=True)

json.dump(res, open(os.path.join(BASE, "results_extremes.json"), "w"), ensure_ascii=False, indent=2)
print("\n已保存 results_extremes.json")
