# -*- coding: utf-8 -*-
"""论文图表生成，输出 tiff + pdf。"""
import sys, io, os, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

# ---------- 全局样式 ----------
plt.rcParams.update({
    "font.sans-serif": ["PingFang SC", "Heiti SC", "Arial Unicode MS", "Hiragino Sans GB"],
    "font.family": "sans-serif",
    "axes.unicode_minus": False,
    "font.size": 8.5,
    "axes.titlesize": 9,
    "axes.labelsize": 8.5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7.5,
    "axes.linewidth": 0.7,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "savefig.dpi": 600,
})

# Okabe-Ito 色盲安全配色
C_PPO, C_GREEDY, C_STATIC, C_QL = "#0072B2", "#E69F00", "#7F7F7F", "#009E73"
C_WARN, C_BURST = "#F0E442", "#D55E00"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RES = BASE_DIR
FIG = os.path.join(os.path.dirname(BASE_DIR), "Figures")


def style_ax(ax, grid=True):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid:
        ax.grid(axis="y", alpha=0.25, linewidth=0.5)
        ax.set_axisbelow(True)


def save(fig, name):
    os.makedirs(FIG, exist_ok=True)
    tmp = os.path.join(FIG, name + "_tmp.png")
    fig.savefig(tmp, dpi=600, facecolor="white")
    try:
        from PIL import Image
        Image.open(tmp).convert("RGB").save(
            os.path.join(FIG, name + ".tiff"), dpi=(600, 600), compression="tiff_lzw")
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    fig.savefig(os.path.join(FIG, name + ".pdf"))
    plt.close(fig)
    print(name, "完成")


# ============ 图 1：模拟环境结构示意 ============
def fig1_env():
    fig = plt.figure(figsize=(170 / 25.4, 112 / 25.4), layout="constrained")
    gs = fig.add_gridspec(2, 1, height_ratios=[1.9, 1.0])
    ax = fig.add_subplot(gs[0])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 62)
    ax.axis("off")

    # ===== 拓扑带（y 22-62）：只有框和线，无任何说明文字 =====
    box = FancyBboxPatch((3, 33), 26, 21, boxstyle="round,pad=0.6",
                         fc="#E6F1FB", ec="#0072B2", lw=1.3)
    ax.add_patch(box)
    ax.text(16, 49, "中央仓库", ha="center", va="center",
            fontsize=11, weight="bold", color="#0C447C")
    ax.text(16, 42, "总库存 =\n期望需求的 65%", ha="center", va="center",
            fontsize=7.5, color="#333333", linespacing=1.5)

    zones = [("宿舍 A 区", 30), ("宿舍 B 区", 30), ("教学区", 20),
             ("餐饮区", 15), ("医疗与隔离区", 25)]
    ys = np.linspace(57, 23, 5)
    for (z, lam), y in zip(zones, ys):
        b = FancyBboxPatch((70, y - 3.4), 28, 6.8, boxstyle="round,pad=0.5",
                           fc="#E8F3EC", ec="#009E73", lw=1.0)
        ax.add_patch(b)
        ax.text(77, y, z, ha="center", va="center", fontsize=8.5, color="#0B5138")
        ax.text(91, y, f"λ={lam}", ha="center", va="center",
                fontsize=7.5, color="#41684F")
        ar = FancyArrowPatch((29.8, 43.5), (69.2, y), arrowstyle="-|>",
                             mutation_scale=10, color="#9A9A9A", lw=0.9)
        ax.add_patch(ar)

    # ===== 信息带（y 2-19）：三行文字，与拓扑带以分隔线隔开，零重叠 =====
    ax.plot([0, 100], [17, 17], color="#CCCCCC", lw=0.7)
    infos = [
        (12.5, "#0072B2", "调运", "每日每区独立选择 0/40/120/240 箱，次日到达；"
                                  "人力成本 = 2 + 0.15 × 运量"),
        (7.5, "#D55E00", "预警", "确认爆发前 2 天晨检预警（三态 0/1/2）；"
                                  "爆发区需求 ×5，持续 10 天，每回合两波"),
        (2.5,  "#666666", "奖励", "r = −( 物资缺口 + 0.3 × 人力成本 )；"
                                  "总库存 = 期望需求的 65%"),
    ]
    for y, c, tag, txt in infos:
        ax.add_patch(Rectangle((2, y - 1.4), 2.6, 2.8, fc=c, ec="none"))
        ax.text(7.5, y, tag, ha="left", va="center", fontsize=7.5,
                weight="bold", color=c)
        ax.text(16, y, txt, ha="left", va="center", fontsize=7, color="#444444")

    # ===== 下：时间轴（直接标注，无浮动图例） =====
    ax2 = fig.add_subplot(gs[1])
    t = np.linspace(0, 30, 300)
    curve = 1 + 1.5 * np.exp(-((t - 15) / 6) ** 2)
    ax2.plot(t, curve, color="#0072B2", lw=1.6)
    ax2.axvspan(6, 8, color=C_WARN, alpha=0.55)
    ax2.axvspan(8, 15, color=C_BURST, alpha=0.14)
    ax2.axvspan(15, 18, color=C_BURST, alpha=0.38)
    ax2.axvspan(18, 25, color=C_BURST, alpha=0.14)
    ax2.axhline(1.0, color="#999999", lw=0.6, ls=":")

    ax2.text(7, 2.62, "预警窗口", ha="center", va="center", fontsize=6.8,
             color="#7a6a00")
    ax2.text(11.5, 2.62, "爆发①", ha="center", va="center", fontsize=6.8,
             color="#a04a10")
    ax2.text(16.5, 2.62, "两波重叠", ha="center", va="center", fontsize=6.8,
             weight="bold", color="#8C2D00")
    ax2.text(21.5, 2.62, "爆发②", ha="center", va="center", fontsize=6.8,
             color="#a04a10")
    ax2.text(2.2, 1.28, "日需求曲线", ha="left", va="center", fontsize=7.5,
             color="#0072B2")

    ax2.set_xlim(0, 30)
    ax2.set_ylim(0.8, 2.78)
    ax2.set_xticks(range(0, 31, 5))
    ax2.set_xlabel("回合时间（天）")
    ax2.set_ylabel("需求倍率")
    style_ax(ax2)

    save(fig, "Fig1 Env")


# ============ 图 2：PPO 训练曲线 ============
def fig2_training():
    curves = [np.load(os.path.join(RES, f"ppo_curve_seed{s}.npz")) for s in range(5)]
    steps = curves[0]["steps"] / 1000.0
    arr = np.array([c["reward"] for c in curves])

    fig, ax = plt.subplots(figsize=(89 / 25.4, 62 / 25.4), layout="constrained")
    mean, sd = arr.mean(axis=0), arr.std(axis=0)
    ax.fill_between(steps, mean - sd, mean + sd, color=C_PPO, alpha=0.18,
                    label="种子间均值 ± SD")
    ax.plot(steps, mean, color=C_PPO, lw=1.8, label="5 种子均值")
    for i in range(5):
        ax.plot(steps, arr[i], color=C_PPO, lw=0.5, alpha=0.3)

    # 起点标记：只画点不写字，避免任何文字压住曲线或置信带
    ax.plot([steps[0]], [arr[:, 0].mean()], marker="o", ms=4.2, mfc="white",
            mec="#444444", mew=1.0, zorder=6)
    ax.set_xlabel("训练步数（×10³）")
    ax.set_ylabel("评估回合累计奖励")
    fig.legend(loc="outside upper center", ncol=2, frameon=False)
    style_ax(ax)
    save(fig, "Fig3 Training")


# ============ 图 3：敏感性分析（双面板） ============
def fig4_sensitivity():
    with open(os.path.join(RES, "results_sensitivity.json"), encoding="utf-8") as f:
        sw = json.load(f)

    names = [("静态均分", C_STATIC, "s", "--"), ("贪婪调配", C_GREEDY, "^", "-."),
             ("PPO", C_PPO, "o", "-")]

    fig, axes = plt.subplots(1, 2, figsize=(170 / 25.4, 62 / 25.4),
                             layout="constrained")
    # --- 面板 1：需求强度 ---
    ax = axes[0]
    d = sw["demand"]
    xs = [x["value"] for x in d]
    for n, c, mk, ls in names:
        ys = [x[n]["shortage"] for x in d]
        ax.plot(xs, ys, marker=mk, ls=ls, color=c, lw=1.5, ms=4.5, label=n)
    ax.axvline(1.0, color="#999999", lw=0.8, ls=":")
    ax.text(1.24, 18100, "较静态缺口\n基准 −10.6%\n2.5× −16.9%", fontsize=7.5,
            color=C_PPO, va="top", linespacing=1.5)
    ax.set_xlabel("需求强度倍率")
    ax.set_ylabel("累计物资缺口（箱）")
    ax.set_title("(a) 需求强度扫描", loc="left", fontsize=8.5)
    style_ax(ax)

    # --- 面板 2：库存倍率 ---
    ax = axes[1]
    d = sw["inventory"]
    xs = [x["value"] for x in d]
    for n, c, mk, ls in names:
        ys = [x[n]["shortage"] for x in d]
        ax.plot(xs, ys, marker=mk, ls=ls, color=c, lw=1.5, ms=4.5, label=n)
    ax.axvline(1.0, color="#999999", lw=0.8, ls=":")
    ax.annotate("静态无法利用\n冗余库存", xy=(1.4, 4136.1), xytext=(1.05, 4650),
                fontsize=7, color="#777777",
                arrowprops=dict(arrowstyle="->", color="#999999", lw=0.7))
    ax.set_xlabel("库存倍率")
    ax.set_ylabel("累计物资缺口（箱）")
    ax.set_title("(b) 库存水平扫描", loc="left", fontsize=8.5)
    style_ax(ax)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=3, frameon=False)
    save(fig, "Fig4 Sensitivity")


# ============ 图 4：监测信号缺失鲁棒性 ============
def fig5_robustness():
    with open(os.path.join(RES, "results_robustness.json"), encoding="utf-8") as f:
        res = json.load(f)
    xs = np.array([r["drop_prob"] * 100 for r in res])

    fig, ax = plt.subplots(figsize=(89 / 25.4, 62 / 25.4), layout="constrained")

    # PPO：5 种子均值 + min-max 范围
    ppo_seeds = np.array([[r[f"PPO_s{i}"]["service"] for r in res] for i in range(5)])
    ax.fill_between(xs, ppo_seeds.min(axis=0), ppo_seeds.max(axis=0),
                    color=C_PPO, alpha=0.15)
    ax.plot(xs, ppo_seeds.mean(axis=0), "o-", color=C_PPO, lw=1.8, ms=5,
            label="PPO（5 种子均值 ± 范围）")

    ax.plot(xs, [r["贪婪调配"]["service"] for r in res], "^--", color=C_GREEDY,
            lw=1.5, ms=5, label="贪婪调配")
    ax.plot(xs, [r["静态均分"]["service"] for r in res], "s:", color=C_STATIC,
            lw=1.5, ms=5, label="静态均分")

    ax.set_xticks(xs)
    ax.set_xlabel("监测信号漏报率（%）")
    ax.set_ylabel("物资保障率")
    ax.set_ylim(0.54, 0.66)
    fig.legend(loc="outside upper center", ncol=3, frameon=False)
    style_ax(ax)
    save(fig, "Fig5 Robustness")


# ============ 备用：主结果柱状图（论文未用，保持可复现） ============
def fig3_main():
    with open(os.path.join(RES, "results_main.json"), encoding="utf-8") as f:
        res = json.load(f)
    names = ["静态均分", "贪婪调配", "Q-learning", "PPO"]
    colors = [C_STATIC, C_GREEDY, C_QL, C_PPO]
    metrics = [("shortage", "累计物资缺口（箱）"),
               ("cost", "累计调运人力成本"),
               ("service", "物资保障率")]
    fig, axes = plt.subplots(1, 3, figsize=(170 / 25.4, 56 / 25.4),
                             layout="constrained")
    for ax, (m, label) in zip(axes, metrics):
        means = [res[n][m]["mean"] for n in names]
        stds = [res[n][m]["std"] for n in names]
        bars = ax.bar(range(4), means, yerr=stds, capsize=2.5,
                      color=colors, width=0.62,
                      error_kw=dict(lw=0.8, ecolor="#444444"))
        ax.set_xticks(range(4))
        ax.set_xticklabels(["静态\n均分", "贪婪\n调配", "Q-\nlearning", "PPO"],
                           fontsize=7.5)
        ax.set_title(label, fontsize=8.5)
        style_ax(ax)
        # 数值标注放在误差条帽上方，避免压线
        top_all = max(mn + st for mn, st in zip(means, stds))
        if m != "service":
            ax.set_ylim(0, top_all * 1.22)
        pad = top_all * 0.025
        for b, mv, st in zip(bars, means, stds):
            ax.text(b.get_x() + b.get_width() / 2, mv + st + pad,
                    f"{mv:.2f}" if m == "service" else f"{mv:.0f}",
                    ha="center", va="bottom", fontsize=7)
        if m == "service":
            ax.set_ylim(0, 0.85)
    save(fig, "Fig2 Main")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="论文图表生成（tiff + pdf）")
    ap.add_argument("--out", default=FIG, help="图输出目录（默认：包内 Figures）")
    ap.add_argument("--data", default=RES, help="数据目录（默认：本目录）")
    args = ap.parse_args()
    FIG, RES = args.out, args.data
    fig1_env()
    fig2_training()
    fig3_main()
    fig4_sensitivity()
    fig5_robustness()
    print("全部图表重制完毕")
