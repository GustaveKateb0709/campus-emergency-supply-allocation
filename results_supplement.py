# -*- coding: utf-8 -*-
"""补充统计：BC 零样本基线 + 策略层面配对检验（生成 results_supplement.json）
运行：python results_supplement.py （需 Train Models/ 下有 bc_policy.zip 与 ppo_seed0-4.zip）
"""
import os, sys, json, math
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
from stable_baselines3 import PPO
from policies import StaticPolicy, GreedyPolicy
from evaluate import evaluate_policy, make_policy_fn

N_EVAL = 100


# ---------- t 分布尾概率（无需 scipy） ----------
def _betacf(a, b, x, itmax=300, eps=3e-16):
    """正则化不完全贝塔函数的连分式展开（Lentz 算法）"""
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < 1e-30:
        d = 1e-30
    d = 1.0 / d
    h = d
    for m in range(1, itmax + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        de = d * c
        h *= de
        if abs(de - 1.0) < eps:
            break
    return h


def _betainc(a, b, x):
    """正则化不完全贝塔函数 I_x(a, b)"""
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
    """t 检验双侧 p 值：P(|T_df| >= |t|) = I_{df/(df+t^2)}(df/2, 1/2)"""
    t = abs(t)
    return _betainc(df / 2.0, 0.5, df / (df + t * t))


def paired_t(g, p):
    """配对 t 检验（正态近似 p 值，n≥100 时误差可忽略）"""
    d = np.array(g) - np.array(p)
    n = len(d)
    md, sd = d.mean(), d.std(ddof=1)
    t = md / (sd / math.sqrt(n))
    pval = 2 * (1 - 0.5 * (1 + math.erf(abs(t) / math.sqrt(2))))
    return {"n": n, "mean_diff": float(md), "sd_diff": float(sd), "t": float(t),
            "p_norm": float(pval),
            "ci95": [float(md - 1.96 * sd / math.sqrt(n)), float(md + 1.96 * sd / math.sqrt(n))]}


def main():
    supp = {}

    # BC 零样本（100 回合，主协议）
    bc_model = PPO.load(os.path.join("Train Models", "bc_policy.zip"), device="cpu")
    s_bc, d_bc = evaluate_policy(lambda obs: bc_model.predict(obs, deterministic=True)[0],
                                 n_episodes=N_EVAL)
    supp["BC_zero_shot"] = {k: {"mean": v[0], "std": v[1]} for k, v in s_bc.items()}

    # 贪婪逐回合
    s_g, d_g = evaluate_policy(make_policy_fn(GreedyPolicy()), n_episodes=N_EVAL)
    supp["Greedy_detail"] = {k: v.tolist() for k, v in d_g.items()}

    # PPO 5 种子逐回合
    ppo_details = {}
    for seed in range(5):
        model = PPO.load(os.path.join("Train Models", f"ppo_seed{seed}"), device="cpu")
        s_p, d_p = evaluate_policy(lambda obs, m=model: m.predict(obs, deterministic=True)[0],
                                   n_episodes=N_EVAL)
        ppo_details[f"seed{seed}"] = {k: v.tolist() for k, v in d_p.items()}
    supp["PPO_per_episode"] = ppo_details

    # 配对检验
    # 注意推断单位：策略层面（种子为聚类）。回合级 pooling 会高估显著性，
    # 此处同时报告两种口径，正文采用策略层面（聚类校正）结论。
    paired = {}
    for seed in range(5):
        paired[f"seed{seed}"] = paired_t(d_g["shortage"], ppo_details[f"seed{seed}"]["shortage"])
    g_all = np.tile(d_g["shortage"], 5)
    p_all = np.concatenate([ppo_details[f"seed{s}"]["shortage"] for s in range(5)])
    pooled = paired_t(g_all, p_all)
    # 策略层面（n=5 种子均值配对）
    seed_means = [np.mean(ppo_details[f"seed{s}"]["shortage"]) for s in range(5)]
    paired["strategy_level"] = {
        "n_seeds": 5,
        "greedy_mean": float(np.mean(d_g["shortage"])),
        "ppo_seed_means": [float(x) for x in seed_means],
        "mean_diff": float(np.mean(seed_means) - np.mean(d_g["shortage"])),
        "sd_diff": float(np.std(seed_means, ddof=1)),
    }
    md, sd = paired["strategy_level"]["mean_diff"], paired["strategy_level"]["sd_diff"]
    paired["strategy_level"]["t"] = float(md / (sd / math.sqrt(5)))
    # n=5 时必须用 t 分布（df=n-1=4）；正态近似会把 p 低估约十倍（0.005 vs 0.050）
    paired["strategy_level"]["p_t"] = float(t_two_sided_p(
        paired["strategy_level"]["t"], df=paired["strategy_level"]["n_seeds"] - 1))
    paired["strategy_level"]["ci95"] = [
        float(md - 2.776 * sd / math.sqrt(5)), float(md + 2.776 * sd / math.sqrt(5))]
    # 上式 2.776 为 t(0.975, df=4) 临界值，仅在 5 个种子时成立
    paired["pooled_round_level_caution"] = pooled  # 回合级 pooling，仅参考
    paired["BC_vs_greedy"] = paired_t(d_g["shortage"], d_bc["shortage"])
    supp["paired_tests"] = paired

    with open("results_supplement.json", "w", encoding="utf-8") as f:
        json.dump(supp, f, ensure_ascii=False, indent=1)
    print("已保存 results_supplement.json")
    print(f"策略层面: 差 {md:.1f} 箱, t={paired['strategy_level']['t']:.2f}, "
          f"p={paired['strategy_level']['p_t']:.3f}（n=5，t 检验 df=4）")
    print(f"回合级 pooling（仅参考，忽略聚类）: {pooled['mean_diff']:.1f} 箱, p={pooled['p_norm']:.3g}")


if __name__ == "__main__":
    main()
