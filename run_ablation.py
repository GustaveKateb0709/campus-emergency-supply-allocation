# -*- coding: utf-8 -*-
"""消融实验：拆开贪婪调配的两个信号加成项（生成 results_ablation.json）

贪婪规则对每个分区按 deficit = max(0, 当日需求率 - 库存) + burst × 需求率 下单：
    burst = 1（预警期）→ 额外补 1 天量，即“预警加成”
    burst = 2（确认爆发）→ 额外补 2 天量，即“确认加成”
本脚本分别关掉其中一项，看累计缺口如何变化，用以判断两项加成各自是正贡献还是过度调运。

    greedy    原策略（两项加成都在）
    no_warn   只关预警加成（flag=1 → 0）
    no_burst  只关确认加成（flag=2 → 0）
    no_signal 两项都关

协议：100 回合（评估种子 1000–1099，与训练种子分离），逐回合配对，t 检验 df=n-1。
运行：python run_ablation.py
"""
import os, sys, json, math
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
from campus_env import N_ZONES
from policies import GreedyPolicy
from evaluate import evaluate_policy, make_policy_fn

N_EVAL = 100
SEED0 = 1000


class _Masked(GreedyPolicy):
    """把指定的爆发标记置 0 后再交给原贪婪规则决策"""

    drop_flags = ()

    def act(self, obs, env=None):
        obs = np.array(obs, dtype=np.float64, copy=True)
        sl = slice(2 * N_ZONES, 3 * N_ZONES)
        flags = obs[sl]
        for f in self.drop_flags:
            flags = np.where(flags == float(f), 0.0, flags)
        obs[sl] = flags
        return super().act(obs, env=env)


class NoWarn(_Masked):
    name = "no_warn"
    drop_flags = (1,)


class NoBurst(_Masked):
    name = "no_burst"
    drop_flags = (2,)


class NoSignal(_Masked):
    name = "no_signal"
    drop_flags = (1, 2)


# ---------- t 分布尾概率（无需 scipy） ----------
def _betacf(a, b, x, itmax=300, eps=3e-16):
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1e-30 if abs(d) < 1e-30 else d
    d = 1.0 / d
    h = d
    for m in range(1, itmax + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1e-30 if abs(d) < 1e-30 else d
        c = 1.0 + aa / c
        c = 1e-30 if abs(c) < 1e-30 else c
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1e-30 if abs(d) < 1e-30 else d
        c = 1.0 + aa / c
        c = 1e-30 if abs(c) < 1e-30 else c
        d = 1.0 / d
        de = d * c
        h *= de
        if abs(de - 1.0) < eps:
            break
    return h


def _betainc(a, b, x):
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lnbt = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
            + a * math.log(x) + b * math.log1p(-x))
    bt = math.exp(lnbt)
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def t_two_sided_p(t, df):
    t = abs(t)
    return _betainc(df / 2.0, 0.5, df / (df + t * t))


def paired(a, b):
    """b 相对 a 的配对差（正值 = b 更差）"""
    d = np.asarray(b) - np.asarray(a)
    n = len(d)
    md, sd = float(d.mean()), float(d.std(ddof=1))
    t = md / (sd / math.sqrt(n))
    se = sd / math.sqrt(n)
    crit = 1.984                                    # t(0.975, df=99)
    return {"n": n, "mean_diff": md, "sd_diff": sd, "t": float(t),
            "p_t": float(t_two_sided_p(t, n - 1)),
            "ci95": [md - crit * se, md + crit * se]}


def main():
    res = {"protocol": {"n_episodes": N_EVAL, "seed0": SEED0}}
    base = None
    variants = [("greedy", GreedyPolicy), ("no_warn", NoWarn),
                ("no_burst", NoBurst), ("no_signal", NoSignal)]
    for key, cls in variants:
        s, d = evaluate_policy(make_policy_fn(cls()), n_episodes=N_EVAL, seed0=SEED0)
        res[key] = {"shortage": [float(s["shortage"][0]), float(s["shortage"][1])],
                    "service": [float(s["service"][0]), float(s["service"][1])]}
        if base is None:
            base = d["shortage"]
        else:
            res[key]["vs_greedy"] = paired(base, d["shortage"])
        print(f"{key:10} 缺口 {s['shortage'][0]:8.2f} ± {s['shortage'][1]:7.2f} | "
              f"保障率 {s['service'][0]:.4f}")

    for k in ("no_warn", "no_burst", "no_signal"):
        v = res[k]["vs_greedy"]
        print(f"  {k} 相对贪婪: {v['mean_diff']:+.2f} 箱, "
              f"t={v['t']:.2f}, p={v['p_t']:.3g}, 95%CI[{v['ci95'][0]:+.1f}, {v['ci95'][1]:+.1f}]")

    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "results_ablation.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print("已保存 results_ablation.json")


if __name__ == "__main__":
    main()
