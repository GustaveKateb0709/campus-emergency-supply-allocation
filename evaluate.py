# -*- coding: utf-8 -*-
"""
统一评估框架：所有策略在相同评估协议下比较
指标：累计物资缺口、累计调运人力成本、缺货天数、物资保障率
协议：固定评估种子序列（与训练种子分离），默认 100 个回合
"""
import numpy as np
from campus_env import CampusSupplyEnv, N_ZONES


def make_policy_fn(policy):
    """规则策略统一接口：obs -> MultiDiscrete 动作（透传 reset 以支持有状态策略）"""
    def fn(obs, info=None):
        return policy.act(obs, env=None)
    fn.reset = policy.reset
    fn.name = getattr(policy, "name", policy.__class__.__name__)
    return fn


def evaluate_policy(policy_fn, n_episodes=100, seed0=1000,
                    env_kwargs=None, reset_each=True):
    """评估策略，返回指标字典 {metric: (mean, std)} 与逐回合明细

    policy_fn: obs -> MultiDiscrete 动作向量；可选属性 .reset() 在每回合开始调用
    """
    env_kwargs = env_kwargs or {}
    env = CampusSupplyEnv(**env_kwargs)
    ep_shortage, ep_cost, ep_stockout, ep_service = [], [], [], []
    for i in range(n_episodes):
        obs, info = env.reset(seed=seed0 + i)
        if reset_each and hasattr(policy_fn, "reset"):
            policy_fn.reset()
        total_shortage, total_cost, total_demand, stockout_days = 0.0, 0.0, 0.0, 0
        done = False
        while not done:
            a = policy_fn(obs)
            obs, r, term, trunc, info = env.step(np.asarray(a, dtype=int).ravel())
            total_shortage += info["shortage"]
            total_cost += info["transport_cost"]
            total_demand += info["demand"]
            stockout_days += int(info["stockout_day"])
            done = term or trunc
        ep_shortage.append(total_shortage)
        ep_cost.append(total_cost)
        ep_stockout.append(stockout_days)
        ep_service.append(1.0 - total_shortage / max(total_demand, 1e-9))
    detail = {"shortage": np.array(ep_shortage), "cost": np.array(ep_cost),
              "stockout_days": np.array(ep_stockout), "service": np.array(ep_service)}
    summary = {k: (float(v.mean()), float(v.std())) for k, v in detail.items()}
    return summary, detail
