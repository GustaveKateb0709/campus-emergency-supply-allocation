# -*- coding: utf-8 -*-
"""
策略行为分析：固定典型爆发场景（宿舍A区，第 8 天爆发），
逐日记录 PPO 与贪婪基线的调运决策与库存轨迹 → results/behavior_trace.json
用于论文"策略行为对比分析"：验证学习策略是否利用预警信号提前预置
"""
import sys, io, os, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from stable_baselines3 import PPO
from campus_env import CampusSupplyEnv, N_ZONES, TRANSFER_OPTIONS, EPISODE_DAYS, BURST_LEN
from policies import StaticPolicy, GreedyPolicy

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = BASE_DIR
MODEL_DIR = os.path.join(BASE_DIR, "Train Models")
SCEN = dict(bursts=[(0, 8), (2, 15)])   # 典型场景：宿舍A区第8天、教学区第15天先后爆发


def trace_policy(policy_name, act_fn, seed=1234):
    env = CampusSupplyEnv()
    obs, _ = env.reset(seed=seed)
    env.bursts = SCEN["bursts"]
    if hasattr(act_fn, "reset"):
        act_fn.reset()
    trace = {"days": [], "transfers": [], "zone_stock": [], "shortage": [],
             "demand": [], "bursts": SCEN["bursts"]}
    done, day = False, 0
    while not done:
        a = np.asarray(act_fn(obs), dtype=int).ravel()
        amounts = [TRANSFER_OPTIONS[a[z]] for z in range(N_ZONES)]
        obs, r, term, trunc, info = env.step(a)
        trace["days"].append(day)
        trace["transfers"].append(amounts)
        trace["zone_stock"].append(env.zone_stock.tolist())
        trace["shortage"].append(info["shortage"])
        trace["demand"].append(info["demand"])
        day += 1
        done = term or trunc
    trace["total_shortage"] = float(np.sum(trace["shortage"]))
    trace["total_cost"] = float(sum((2 + 0.15 * sum(t)) for t in trace["transfers"]))
    print(f"{policy_name}: 总缺口 {trace['total_shortage']:.0f} 总成本 {trace['total_cost']:.0f}")
    return trace


def main():
    from policies import StaticPolicy, GreedyPolicy
    from evaluate import make_policy_fn
    out = {}
    sp, gp = StaticPolicy(), GreedyPolicy()
    out["静态均分"] = trace_policy("静态均分", make_policy_fn(sp))
    out["贪婪调配"] = trace_policy("贪婪调配", make_policy_fn(gp))

    model = PPO.load(os.path.join(MODEL_DIR, "ppo_seed0"))
    out["PPO"] = trace_policy("PPO", lambda obs: model.predict(obs, deterministic=True)[0])

    # 关键行为指标：预警期与爆发首日的调运响应
    for name in out:
        tr = out[name]
        metrics = {}
        for bz, bs in SCEN["bursts"]:
            warn = sum(tr["transfers"][d][bz] for d in range(max(0, bs - 2), bs))
            first3 = sum(tr["transfers"][d][bz] for d in range(bs, min(bs + 3, EPISODE_DAYS)))
            metrics[f"波次{bs}日爆区{bz}_预警期"] = float(warn)
            metrics[f"波次{bs}日爆区{bz}_爆发前3日"] = float(first3)
        out[name].update(metrics)
        print(f"  {name}: " + " | ".join(f"{k}={v:.0f}" for k, v in metrics.items()))

    with open(os.path.join(OUT_DIR, "behavior_trace.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("已保存 behavior_trace.json")


if __name__ == "__main__":
    main()
