# -*- coding: utf-8 -*-
"""校园应急物资调配模拟环境（Gymnasium 接口）。"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces

# ---------- 场景常量 ----------
ZONES = ["宿舍A区", "宿舍B区", "教学区", "餐饮区", "医疗与隔离区"]
N_ZONES = 5
EPISODE_DAYS = 30                       # 每回合模拟一轮 30 天疫情周期
BASE_DEMAND = np.array([30.0, 30.0, 20.0, 15.0, 25.0])   # 各分区基础日均需求（单位：箱）
BURST_MULT = 5.0                        # 爆发期需求倍数
BURST_LEN = 10                          # 爆发持续天数
BURST_WARNING = 2                       # 监测预警提前量（天）：信号早于确认爆发出现
BURST_N = 2                             # 每回合爆发波次数（多波次疫情冲击）
TRANSFER_OPTIONS = [0.0, 40.0, 120.0, 240.0]             # 调运量档位（单位：箱/日/区）
TRANSPORT_FIXED = 2.0                   # 每次调运固定人力成本
TRANSPORT_UNIT = 0.15                   # 每箱变动人力成本（调运、装卸、押运）
INVENTORY_COVERAGE = 0.65               # 总库存对期望总需求的覆盖比例（资源稀缺）


def demand_curve(t, peak_day=15, width=6.0):
    """疫情日需求曲线：单峰型，中期达到高峰（相对基础需求最高 2.5 倍）"""
    return 1.0 + 1.5 * np.exp(-((t - peak_day) / width) ** 2)


class CampusSupplyEnv(gym.Env):
    """Gymnasium 兼容的校园应急物资调配环境

    动作空间：MultiDiscrete([4]*5) —— 每个分区独立选择当日调运量档位
    观测空间（17 维 float32）：
        [0:5]   各分区库存 / 100
        [5:10]  各分区当日需求强度（泊松率）/ 50
        [10:15] 各分区爆发标记（0/1）
        [15]    中央仓库库存 / 1000
        [16]    回合进度 t / 30
    """

    metadata = {"render_modes": []}

    def __init__(self, demand_scale=1.0, inventory_scale=1.0,
                 transport_weight=0.3, **kwargs):
        super().__init__()
        self.demand_scale = demand_scale
        self.inventory_scale = inventory_scale
        self.transport_weight = transport_weight

        # 总库存 = 覆盖率 × 蒙特卡洛估计的期望总需求（含疫情曲线与爆发冲击的期望）
        # 使 inventory_scale=1 时实际覆盖率恒为 INVENTORY_COVERAGE，保证不同
        # demand_scale 下的实验可比性
        expected_demand = self._estimate_expected_demand()
        self.total_inventory = float(expected_demand * INVENTORY_COVERAGE
                                     * inventory_scale)

        self.action_space = spaces.MultiDiscrete([len(TRANSFER_OPTIONS)] * N_ZONES)
        obs_dim = 3 * N_ZONES + 2
        self.observation_space = spaces.Box(low=0.0, high=np.inf,
                                            shape=(obs_dim,), dtype=np.float32)

    def _estimate_expected_demand(self, n_sim=300, seed=12345):
        """蒙特卡洛估计期望总需求（含曲线放大与多波次爆发冲击期望）"""
        rng = np.random.default_rng(seed)
        totals = []
        lam0 = BASE_DEMAND * self.demand_scale
        for _ in range(n_sim):
            zs = rng.choice(N_ZONES, size=BURST_N, replace=False)
            starts = sorted(rng.integers(5, 10, size=BURST_N) + np.arange(BURST_N) * 5)
            bursts = [(int(z), int(s)) for z, s in zip(zs, starts)]
            total = 0.0
            for t in range(EPISODE_DAYS):
                lam = lam0 * demand_curve(t)
                for bz, bs in bursts:
                    if bs <= t < bs + BURST_LEN:
                        lam[bz] *= BURST_MULT
                total += float(lam.sum())
            totals.append(total)
        return float(np.mean(totals))

    # ---------- 内部工具 ----------
    def _get_obs(self):
        obs = np.concatenate([
            self.zone_stock / 100.0,
            self.demand_rate / 50.0,
            self.burst_flag.astype(np.float64),
            [self.central_stock / 1000.0],
            [self.t / EPISODE_DAYS],
        ])
        return obs.astype(np.float32)

    def _sample_demand_rate(self):
        """当日需求率：基础需求 × 日曲线 × 需求倍率 × 爆发冲击

        爆发标记（监测预警信号）三态：0=无信号，1=预警期（确认爆发前
        BURST_WARNING 天，对应晨检聚集性预警），2=确认爆发中。
        """
        lam = BASE_DEMAND * demand_curve(self.t) * self.demand_scale
        self.burst_flag[:] = 0.0
        for bz, bs in self.bursts:
            if bs - BURST_WARNING <= self.t < bs + BURST_LEN:
                self.burst_flag[bz] = 1.0 if self.t < bs else 2.0
                if self.t >= bs:
                    lam[bz] *= BURST_MULT
        self.demand_rate = lam
        return lam

    # ---------- Gymnasium 接口 ----------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.rng = np.random.default_rng(seed)
        self.t = 0
        self.zone_stock = np.zeros(N_ZONES)      # 开局各分区无库存
        self.central_stock = float(self.total_inventory)
        self.pending = []                         # 在途调运 [(到达日, 分区, 数量)]
        # 多波次爆发：BURST_N 个不同分区先后爆发
        bz = self.rng.choice(N_ZONES, size=BURST_N, replace=False)
        starts = sorted(self.rng.integers(5, 10, size=BURST_N)
                        + np.arange(BURST_N) * 5)
        self.bursts = [(int(z), int(s)) for z, s in zip(bz, starts)]
        self.burst_flag = np.zeros(N_ZONES)
        self.demand_rate = BASE_DEMAND.copy()
        return self._get_obs(), {}

    def step(self, action):
        # 1) 在途调运到达（转运延迟 1 天：前一日发出，当日晨间入账）
        remaining = []
        for (day, z, amt) in self.pending:
            if day <= self.t:
                self.zone_stock[z] += amt
            else:
                remaining.append((day, z, amt))
        self.pending = remaining

        # 2) 当日需求发生与满足
        lam = self._sample_demand_rate()
        demand = self.rng.poisson(lam).astype(np.float64)
        served = np.minimum(self.zone_stock, demand)
        shortage = demand - served
        self.zone_stock -= served

        # 3) 调运决策（当日发出，次日晨到达）
        action = np.asarray(action, dtype=int).ravel()
        transport_cost = 0.0
        for z in range(N_ZONES):
            amount = TRANSFER_OPTIONS[action[z]]
            if amount > 0:
                amount = min(amount, self.central_stock)
                if amount > 0:
                    self.central_stock -= amount
                    self.pending.append((self.t + 1, z, amount))
                    transport_cost += TRANSPORT_FIXED + TRANSPORT_UNIT * amount

        # 4) 奖励与推进
        reward = -(float(shortage.sum()) + self.transport_weight * transport_cost)
        self.t += 1
        terminated = self.t >= EPISODE_DAYS
        truncated = False
        info = {
            "shortage": float(shortage.sum()),
            "demand": float(demand.sum()),
            "transport_cost": float(transport_cost),
            "stockout_day": bool(shortage.sum() > 0),
        }
        return self._get_obs(), reward, terminated, truncated, info
