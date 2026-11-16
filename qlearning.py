# -*- coding: utf-8 -*-
"""分解式表格 Q-learning：每个分区独立维护一张 Q 表，用于与 PPO 对比。"""
import numpy as np
from campus_env import N_ZONES, EPISODE_DAYS

CENTRAL_BINS = np.array([400.0, 1200.0])        # 中央库存分档阈值
PHASE_BINS = np.array([10, 20])                  # 疫情阶段分界（天）
ZSTOCK_BINS = np.array([150.0, 400.0])           # 分区库存分档阈值
N_ACT = 4                                        # 每区动作：0/40/120/240


def zone_state(z, obs):
    """分区 z 的离散化状态"""
    burst = int(obs[2 * N_ZONES + z] > 0)
    zstock = int(np.digitize(obs[z] * 100.0, ZSTOCK_BINS))
    central = int(np.digitize(obs[15] * 1000.0, CENTRAL_BINS))
    phase = int(np.digitize(obs[16] * EPISODE_DAYS, PHASE_BINS))
    return (burst, zstock, central, phase)


class DecomposedQLearner:
    """分解式多目标表格 Q-learning"""

    def __init__(self, alpha=0.1, gamma=0.99, epsilon=1.0,
                 eps_min=0.05, eps_decay=0.9994, seed=0):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = eps_min
        self.eps_decay = eps_decay
        self.rng = np.random.default_rng(seed)
        self.q = [dict() for _ in range(N_ZONES)]      # q[z][state] -> np.array(4)

    def _qrow(self, z, s):
        if s not in self.q[z]:
            self.q[z][s] = np.zeros(N_ACT)
        return self.q[z][s]

    def act(self, obs, explore=False):
        actions = np.zeros(N_ZONES, dtype=int)
        for z in range(N_ZONES):
            s = zone_state(z, obs)
            if explore and self.rng.random() < self.epsilon:
                actions[z] = int(self.rng.integers(N_ACT))
            else:
                actions[z] = int(np.argmax(self._qrow(z, s)))
        return actions

    def learn_step(self, obs, actions, reward, next_obs, done):
        for z in range(N_ZONES):
            s, s2 = zone_state(z, obs), zone_state(z, next_obs)
            q = self._qrow(z, s)
            target = reward + (0.0 if done else self.gamma * np.max(self._qrow(z, s2)))
            q[actions[z]] += self.alpha * (target - q[actions[z]])

    def decay_eps(self):
        self.epsilon = max(self.eps_min, self.epsilon * self.eps_decay)


def train_qlearning(env_fn, episodes=10000, seed=0, log_every=2000):
    """在训练环境中训练分解式 Q-learning，返回 agent 与学习曲线"""
    agent = DecomposedQLearner(seed=seed)
    curve = []
    for ep in range(episodes):
        env = env_fn()
        obs, _ = env.reset(seed=ep + 20000)
        done = False
        ep_reward = 0.0
        while not done:
            a = agent.act(obs, explore=True)
            next_obs, r, term, trunc, _ = env.step(a)
            agent.learn_step(obs, a, r, next_obs, term)
            obs = next_obs
            ep_reward += r
            done = term or trunc
        agent.decay_eps()
        curve.append(ep_reward)
        if (ep + 1) % log_every == 0:
            recent = float(np.mean(curve[-200:]))
            print(f"  [Q-learning] ep {ep+1}/{episodes} 近200回合奖励均值 {recent:.1f} eps={agent.epsilon:.3f}")
    return agent, curve
