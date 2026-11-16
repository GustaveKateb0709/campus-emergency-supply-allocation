# -*- coding: utf-8 -*-
"""单因素敏感性分析：需求强度 / 库存水平 / 人力成本权重。"""
import sys, io, os, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from stable_baselines3 import PPO
from campus_env import CampusSupplyEnv
from policies import StaticPolicy, GreedyPolicy
from evaluate import evaluate_policy, make_policy_fn

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = BASE_DIR
MODEL_DIR = os.path.join(BASE_DIR, "Train Models")
N_EVAL = 60
SEEDS = [0, 1, 2, 3, 4]


def ppo_policy_fn(seed):
    model = PPO.load(os.path.join(MODEL_DIR, f"ppo_seed{seed}"))
    return lambda obs: model.predict(obs, deterministic=True)[0]


def eval_all(env_kwargs):
    """给定环境参数，返回 {策略名: 指标均值字典}"""
    out = {}
    for policy in [StaticPolicy(), GreedyPolicy()]:
        s, _ = evaluate_policy(make_policy_fn(policy), n_episodes=N_EVAL, env_kwargs=env_kwargs)
        out[policy.name] = {k: v[0] for k, v in s.items()}
    ppo_seeds = []
    for seed in SEEDS:
        s, _ = evaluate_policy(ppo_policy_fn(seed), n_episodes=N_EVAL, env_kwargs=env_kwargs)
        ppo_seeds.append({k: v[0] for k, v in s.items()})
    out["PPO"] = {k: float(np.mean([x[k] for x in ppo_seeds])) for k in ppo_seeds[0]}
    out["PPO_std_across_seeds"] = {k: float(np.std([x[k] for x in ppo_seeds])) for k in ppo_seeds[0]}
    return out


def main():
    sweep = {"demand": [], "inventory": [], "transport": []}

    print("== 维度1：需求强度 ==")
    for ds in [0.5, 0.75, 1.0, 1.5, 2.0, 2.5]:
        r = eval_all(dict(demand_scale=ds))
        sweep["demand"].append({"value": ds, **r})
        print(f"  ds={ds}: 缺口 PPO {r['PPO']['shortage']:.0f} vs 贪婪 {r['贪婪调配']['shortage']:.0f} vs 静态 {r['静态均分']['shortage']:.0f}")

    print("== 维度2：库存倍率 ==")
    for iv in [0.6, 0.8, 1.0, 1.2, 1.4]:
        r = eval_all(dict(inventory_scale=iv))
        sweep["inventory"].append({"value": iv, **r})
        print(f"  iv={iv}: 缺口 PPO {r['PPO']['shortage']:.0f} vs 贪婪 {r['贪婪调配']['shortage']:.0f} vs 静态 {r['静态均分']['shortage']:.0f}")

    print("== 维度3：人力成本权重 ==")
    for tw in [0.1, 0.3, 0.5]:
        r = eval_all(dict(transport_weight=tw))
        sweep["transport"].append({"value": tw, **r})
        print(f"  tw={tw}: 成本 PPO {r['PPO']['cost']:.0f} vs 贪婪 {r['贪婪调配']['cost']:.0f} | 缺口 PPO {r['PPO']['shortage']:.0f} vs 贪婪 {r['贪婪调配']['shortage']:.0f}")

    with open(os.path.join(OUT_DIR, "results_sensitivity.json"), "w", encoding="utf-8") as f:
        json.dump(sweep, f, ensure_ascii=False, indent=2)
    print("已保存 results_sensitivity.json")


if __name__ == "__main__":
    main()
