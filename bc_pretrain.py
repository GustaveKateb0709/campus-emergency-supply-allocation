# -*- coding: utf-8 -*-
"""
行为克隆（BC）预训练：从贪婪规则基线采集决策轨迹，监督预训练 PPO 策略网络。
用法：python bc_pretrain.py  → results/models/bc_policy.zip
"""
import sys, io, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from torch.optim import Adam
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from campus_env import CampusSupplyEnv
from policies import GreedyPolicy
from train_ppo import make_env, MODEL_DIR

N_EPISODES = 2000
EPOCHS = 35
BATCH = 1024


def collect_greedy(n_episodes=N_EPISODES):
    env = CampusSupplyEnv()
    policy = GreedyPolicy()
    obs_list, act_list = [], []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=50_000 + ep)
        done = False
        while not done:
            a = policy.act(obs)
            obs_list.append(obs.copy())
            act_list.append(np.asarray(a, dtype=np.int64))
            obs, r, term, trunc, info = env.step(a)
            done = term or trunc
    obs_arr = np.array(obs_list, dtype=np.float32)
    act_arr = np.array(act_list, dtype=np.int64)
    print(f"采集完成：{len(obs_arr)} 条贪婪决策轨迹（{n_episodes} 回合）")
    return obs_arr, act_arr


def pretrain(obs_arr, act_arr):
    model = PPO("MlpPolicy", DummyVecEnv([make_env(0)]), seed=0, verbose=0,
                policy_kwargs=dict(net_arch=[64, 64]))
    policy = model.policy
    policy.train()
    opt = Adam(policy.parameters(), lr=3e-4)
    obs_t = torch.tensor(obs_arr)
    act_t = torch.tensor(act_arr)
    n = len(obs_t)
    for epoch in range(EPOCHS):
        perm = torch.randperm(n)
        total_loss, n_batches = 0.0, 0
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            dist = policy.get_distribution(obs_t[idx])
            log_prob = dist.log_prob(act_t[idx])       # (N, 5) 各头对数概率
            loss = -log_prob.mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += float(loss)
            n_batches += 1
        print(f"  BC epoch {epoch+1}/{EPOCHS} 平均损失 {-total_loss/n_batches:.4f}")
    return model


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    obs_arr, act_arr = collect_greedy()
    model = pretrain(obs_arr, act_arr)

    # BC 策略的零样本表现
    from evaluate import evaluate_policy
    fn = lambda obs: model.predict(obs, deterministic=True)[0]
    s, _ = evaluate_policy(fn, n_episodes=50)
    print(f"BC 预训练策略（零样本）: 缺口 {s['shortage'][0]:.0f}±{s['shortage'][1]:.0f} | 成本 {s['cost'][0]:.0f} | 保障率 {s['service'][0]:.3f}")

    model.save(os.path.join(MODEL_DIR, "bc_policy"))
    print("已保存 bc_policy.zip")


if __name__ == "__main__":
    main()
