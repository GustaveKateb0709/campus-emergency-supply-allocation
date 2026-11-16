# -*- coding: utf-8 -*-
"""两种规则基线：静态均分预置与贪婪调配。"""
import numpy as np
from campus_env import ZONES, N_ZONES, TRANSFER_OPTIONS, EPISODE_DAYS, BASE_DEMAND, TRANSPORT_FIXED


def _action_index(zone, amount):
    """将 (分区, 调运量) 转成 MultiDiscrete 动作向量（仅该分区非零）"""
    act = np.zeros(N_ZONES, dtype=int)
    act[zone] = TRANSFER_OPTIONS.index(amount)
    return act


class StaticPolicy:
    """静态均分预置：回合开始 5 天内向各分区均铺库存（每日每区 240 箱），
    此后整个回合不再调配（对应"疫情前一次性预置、缺乏动态调度"的传统模式）"""
    name = "静态均分"

    def __init__(self):
        self.day = 0

    def reset(self):
        self.day = 0

    def act(self, obs, env=None):
        if self.day < 5:
            a = np.full(N_ZONES, TRANSFER_OPTIONS.index(240.0), dtype=int)
        else:
            a = np.zeros(N_ZONES, dtype=int)
        self.day += 1
        return a


class GreedyPolicy:
    """贪婪就近调配：对每个分区计算"次日需求率 × 覆盖天数 - 当前库存"，
    缺多少补多少（受调运档位限制），不预留中央库存（短视）"""
    name = "贪婪调配"

    def __init__(self):
        pass

    def reset(self):
        pass

    def act(self, obs, env=None):
        stock = obs[0:N_ZONES] * 100.0
        rate = obs[N_ZONES:2 * N_ZONES] * 50.0          # 当日需求率
        central = obs[15] * 1000.0
        burst = obs[2 * N_ZONES:3 * N_ZONES]

        act = np.zeros(N_ZONES, dtype=int)
        need = rate * 1.0                                # 按当日需求率 1 天前瞻
        deficit = np.maximum(0.0, need - stock)
        # 信号响应：预警期(flag=1)额外补 1 天量，确认爆发(flag=2)额外补 2 天量
        deficit = deficit + burst * rate
        order = np.argsort(-deficit)                     # 缺口大的优先
        remaining_central = central
        for z in order:
            if deficit[z] <= 0 or remaining_central <= TRANSPORT_FIXED:
                continue
            # 选择不超过缺口的最大调运档位
            amount = 0.0
            for opt in TRANSFER_OPTIONS[1:]:
                if opt <= deficit[z] + 1e-9 and opt <= remaining_central:
                    amount = opt
            if amount == 0.0 and deficit[z] > 0 and remaining_central >= TRANSFER_OPTIONS[1]:
                amount = TRANSFER_OPTIONS[1]             # 最小档兜底（贪婪小批量倾向）
            if amount > 0:
                act[z] = TRANSFER_OPTIONS.index(amount)
                remaining_central -= amount
        return act
