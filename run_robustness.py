# -*- coding: utf-8 -*-
"""监测信号漏报下的鲁棒性评估（不重训，评估时对信号加噪）。"""
import sys, io, os, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from stable_baselines3 import PPO
from campus_env import CampusSupplyEnv, N_ZONES
from policies import StaticPolicy, GreedyPolicy
from evaluate import evaluate_policy, make_policy_fn

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = BASE_DIR
MODEL_DIR = os.path.join(BASE_DIR, "Train Models")
N_EVAL = 60
DROP_PROBS = [0.0, 0.15, 0.30, 0.50]


def noisy(fn, drop_prob, seed0=777_000):
    """对 obs 的爆发预警信号按概率丢弃（每步每区独立）"""
    rng = np.random.default_rng(seed0)

    def wrapper(obs):
        if drop_prob <= 0:
            return fn(obs)
        obs2 = obs.copy()
        keep = rng.random(N_ZONES) >= drop_prob
        obs2[2 * N_ZONES:3 * N_ZONES] = obs2[2 * N_ZONES:3 * N_ZONES] * keep
        return fn(obs2)
    wrapper.reset = getattr(fn, "reset", lambda: None)
    wrapper.name = getattr(fn, "name", "?")
    return wrapper


def main():
    results = []
    for p in DROP_PROBS:
        row = {"drop_prob": p}
        for policy in [StaticPolicy(), GreedyPolicy()]:
            fn = noisy(make_policy_fn(policy), p)
            s, _ = evaluate_policy(fn, n_episodes=N_EVAL)
            row[policy.name] = {"service": s["service"][0], "shortage": s["shortage"][0]}
        for seed in [0, 1, 2, 3, 4]:
            model = PPO.load(os.path.join(MODEL_DIR, f"ppo_seed{seed}"))
            fn = noisy(lambda obs: model.predict(obs, deterministic=True)[0], p)
            s, _ = evaluate_policy(fn, n_episodes=N_EVAL)
            row[f"PPO_s{seed}"] = {"service": s["service"][0], "shortage": s["shortage"][0]}
        ppo_service = np.mean([row[f"PPO_s{i}"]["service"] for i in range(5)])
        ppo_short = np.mean([row[f"PPO_s{i}"]["shortage"] for i in range(5)])
        row["PPO"] = {"service": float(ppo_service), "shortage": float(ppo_short)}
        results.append(row)
        print(f"漏报率 {p:.0%}: 保障率 静态 {row['静态均分']['service']:.3f} | "
              f"贪婪 {row['贪婪调配']['service']:.3f} | PPO {ppo_service:.3f}")

    with open(os.path.join(OUT_DIR, "results_robustness.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("已保存 results_robustness.json")


if __name__ == "__main__":
    main()
