# -*- coding: utf-8 -*-
"""
PPO 主实验训练脚本（Stable-Baselines3）
5 个随机种子 × 350k 步，8 个串行向量化环境（DummyVecEnv），定期评估记录学习曲线
"""
import sys, io, os, json, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from campus_env import CampusSupplyEnv

SEEDS = [0, 1, 2, 3, 4]
TOTAL_STEPS = 350_000
N_ENVS = 8
EVAL_EVERY = 10_000
EVAL_EPS = 20
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = BASE_DIR
MODEL_DIR = os.path.join(BASE_DIR, "Train Models")
BC_PATH = os.path.join(MODEL_DIR, "bc_policy.zip")


class RewardScale(gym.Wrapper):
    """奖励缩放（不影响 info 中的原始指标），稳定值函数估计"""

    def __init__(self, env, scale=0.02):
        super().__init__(env)
        self.scale = scale

    def step(self, action):
        obs, r, term, trunc, info = self.env.step(action)
        return obs, r * self.scale, term, trunc, info


def linear_schedule(initial_lr):
    def f(progress):
        return initial_lr * progress
    return f


class SeededEnv(gym.Wrapper):
    """为每个训练回合分配固定且可复现的回合种子，与评估种子区间（1000-1099）完全隔离"""

    def __init__(self, env, seed_base):
        super().__init__(env)
        self.seed_base = seed_base
        self._ep = 0

    def reset(self, **kwargs):
        kwargs.pop("seed", None)
        self._ep += 1
        return self.env.reset(seed=self.seed_base + self._ep, **kwargs)


def make_env(seed, rank=0):
    def _f():
        # 训练回合种子：1_000_000 起，按训练种子与向量化环境编号分段，每段 10000 个回合
        base = 1_000_000 + seed * 100_000 + rank * 10_000
        return SeededEnv(RewardScale(CampusSupplyEnv(), scale=0.02), seed_base=base)
    return _f


def eval_model(model, n_eps=20, seed0=90_000):
    env = CampusSupplyEnv()
    rewards, shortages = [], []
    for i in range(n_eps):
        obs, _ = env.reset(seed=seed0 + i)
        done, ep_r, ep_short = False, 0.0, 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(action)
            ep_r += r
            ep_short += info["shortage"]
            done = term or trunc
        rewards.append(ep_r)
        shortages.append(ep_short)
    return float(np.mean(rewards)), float(np.mean(shortages))


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    t0 = time.time()
    for seed in SEEDS:
        print(f"\n===== PPO 训练 seed={seed} =====")
        env_fns = [make_env(seed, i) for i in range(N_ENVS)]
        venv = VecMonitor(DummyVecEnv(env_fns))
        model = PPO("MlpPolicy", venv, seed=seed, verbose=0,
                    learning_rate=linear_schedule(5e-4), n_steps=512, batch_size=512,
                    gamma=0.99, gae_lambda=0.95, ent_coef=0.003,
                    policy_kwargs=dict(net_arch=[64, 64]))
        if os.path.exists(BC_PATH):
            bc_model = PPO.load(BC_PATH, device="cpu")
            model.policy.load_state_dict(bc_model.policy.state_dict())
            print(f"  已从行为克隆权重热启动（{BC_PATH}）")
        curve_steps, curve_reward, curve_shortage = [], [], []
        step = 0
        while step < TOTAL_STEPS:
            model.learn(total_timesteps=EVAL_EVERY, reset_num_timesteps=False, progress_bar=False)
            step += EVAL_EVERY
            mean_r, mean_short = eval_model(model, EVAL_EPS)
            curve_steps.append(step)
            curve_reward.append(mean_r)
            curve_shortage.append(mean_short)
            print(f"  step {step:>7d} | 评估奖励 {mean_r:>9.1f} | 缺口 {mean_short:>7.1f} | {time.time()-t0:.0f}s")
        np.savez(os.path.join(OUT_DIR, f"ppo_curve_seed{seed}.npz"),
                 steps=np.array(curve_steps), reward=np.array(curve_reward),
                 shortage=np.array(curve_shortage))
        model.save(os.path.join(MODEL_DIR, f"ppo_seed{seed}"))
        del model, venv
    print(f"\n全部训练完成，总耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
