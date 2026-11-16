# -*- coding: utf-8 -*-
"""
主实验汇总：4 策略 × 100 评估回合 → results_main.json
策略：静态均分 / 贪婪调配 / 分解式 Q-learning / PPO（5 种子集成评估）
"""
import sys, io, os, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from stable_baselines3 import PPO
from campus_env import CampusSupplyEnv
from policies import StaticPolicy, GreedyPolicy
from evaluate import evaluate_policy, make_policy_fn
from qlearning import train_qlearning

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = BASE_DIR
MODEL_DIR = os.path.join(BASE_DIR, "Train Models")
N_EVAL = 100
SEEDS = [0, 1, 2, 3, 4]


def ppo_policy_fn(seed):
    model = PPO.load(os.path.join(MODEL_DIR, f"ppo_seed{seed}"))
    return lambda obs: model.predict(obs, deterministic=True)[0]


def main():
    results = {}

    # 规则基线
    for policy in [StaticPolicy(), GreedyPolicy()]:
        s, _ = evaluate_policy(make_policy_fn(policy), n_episodes=N_EVAL)
        results[policy.name] = {k: {"mean": v[0], "std": v[1]} for k, v in s.items()}
        print(f"{policy.name}: " + " | ".join(f"{k}={v[0]:.1f}±{v[1]:.1f}" for k, v in s.items()))

    # Q-learning（固定训练配置，种子 0；另跑 2 个种子评估训练稳定性）
    ql_by_seed = []
    for qs in [0, 1, 2]:
        agent, _ = train_qlearning(lambda: CampusSupplyEnv(), episodes=10000, seed=qs, log_every=10**9)
        agent.epsilon = 0.0
        fn = lambda obs: agent.act(obs)
        fn.reset = lambda: None
        fn.name = "Q-learning"
        s, _ = evaluate_policy(fn, n_episodes=N_EVAL)
        ql_by_seed.append(s)
        print(f"Q-learning(seed{qs}): 缺口 {s['shortage'][0]:.1f}±{s['shortage'][1]:.1f} 保障率 {s['service'][0]:.3f}")
    ql_summary = {k: (float(np.mean([x[k][0] for x in ql_by_seed])),
                      float(np.std([x[k][0] for x in ql_by_seed]))) for k in ql_by_seed[0]}
    results["Q-learning"] = {k: {"mean": v[0], "std": v[1]} for k, v in ql_summary.items()}

    # PPO：5 种子分别评估，报告种子间均值与标准差
    ppo_by_seed = []
    for seed in SEEDS:
        s, _ = evaluate_policy(ppo_policy_fn(seed), n_episodes=N_EVAL)
        ppo_by_seed.append(s)
        print(f"PPO(seed{seed}): 缺口 {s['shortage'][0]:.1f}±{s['shortage'][1]:.1f} 保障率 {s['service'][0]:.3f}")
    ppo_summary = {k: (float(np.mean([x[k][0] for x in ppo_by_seed])),
                       float(np.std([x[k][0] for x in ppo_by_seed]))) for k in ppo_by_seed[0]}
    results["PPO"] = {k: {"mean": v[0], "std": v[1]} for k, v in ppo_summary.items()}
    print(f"PPO(5种子汇总): " + " | ".join(f"{k}={v[0]:.1f}±{v[1]:.1f}" for k, v in ppo_summary.items()))

    with open(os.path.join(OUT_DIR, "results_main.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("已保存 results_main.json")


if __name__ == "__main__":
    main()
