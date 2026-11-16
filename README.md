# 校园突发公共卫生事件应急物资调配 DRL 模拟研究——复现说明

本仓库是论文《高校应急物资动态调配的策略比较：规则启发式与深度强化学习》的复现材料。

## 环境

- Python 3.13（PPO 训练与评估 3.13.12，行为克隆预训练 3.13.14）；Stable-Baselines3 2.9.2、
  PyTorch 2.15.0、Gymnasium 1.4.0、NumPy 2.5.3（完整记录见 `Train Models/*.zip` 内的 `system_info.txt`）。
  依赖：numpy、gymnasium、matplotlib、pandas、stable-baselines3（含 PyTorch CPU 版）

```bash
pip install numpy gymnasium matplotlib pandas stable-baselines3
```

## 复现流程（按顺序）

```bash
python train_ppo.py          # 1. PPO 训练：5 种子 × 350k 步（CPU）
python run_main.py           # 2. 主实验：4 策略 × 100 评估回合 → results_main.json（本目录）
python run_sensitivity.py    # 3. 敏感性分析：3 维度单因素扫描 → results_sensitivity.json（本目录）
python make_tables.py        # 4. 生成论文表格 → 上级目录 Tables/*.csv（--out 可指定）
python make_figures.py       # 5. 生成 5 张论文图表（Fig1–Fig5）→ 上级目录 Figures/*.png|pdf
python run_ablation.py       # 6. 消融：拆分贪婪规则的预警/确认加成 → results_ablation.json（本目录）
```

## 文件说明

| 文件 | 内容 |
|------|------|
| campus_env.py | Gymnasium 模拟环境（5 分区 + 中央仓库；泊松需求 × 疫情曲线 × 爆发冲击；转运延迟 1 天；奖励 = -(缺口 + w·人力成本)） |
| policies.py | 规则基线：静态均分预置、贪婪就近调配 |
| qlearning.py | 分解式表格 Q-learning（每区独立动作值函数，54 状态 × 4 动作/区） |
| evaluate.py | 统一评估协议（固定评估种子，指标：缺口/成本/保障率/缺货天数） |
| train_ppo.py | PPO 训练主脚本 |
| run_main.py / run_sensitivity.py | 主实验与敏感性实验 |
| run_robustness.py / run_ablation.py | 信号漏报鲁棒性；拆分预警/确认两项加成的消融实验 |
| make_tables.py / make_figures.py | 表格与图表 |

## 关键设定

- 覆盖率 0.65（总库存 = 0.65 × 蒙特卡洛期望总需求，需求变化时自动重标定）
- 爆发：每回合随机 2 个不同分区先后爆发（首波第 5—9 天、次波第 10—14 天），需求 ×5，持续 10 天；监测预警提前 2 天（晨检信号）
- 调运档位：每区每日独立 0/40/120/240 箱；转运次日到达
- 成本：每次调运 固定 2 + 0.15×数量；奖励权重 w 默认 0.3
- 评估：训练/评估种子完全分离。评估回合种子 1000–1099；PPO 训练 5 种子（策略网络种子 0–4），Q-learning 训练 3 种子（回合种子 20000–29999，与评估种子不重叠）。PPO 报告 5 种子间均值±标准差，Q-learning 报告 3 种子间均值±标准差
