# -*- coding: utf-8 -*-
"""生成论文表格（CSV，供论文引用）：主结果对比表 + 敏感性汇总表"""
import sys, io, os, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RES = BASE_DIR
TAB = os.path.join(os.path.dirname(BASE_DIR), "Tables")


def main():
    import argparse
    ap = argparse.ArgumentParser(description="论文表格生成（CSV）")
    ap.add_argument("--out", default=TAB, help="表输出目录（默认：包内 Tables）")
    ap.add_argument("--data", default=RES, help="数据目录（默认：本目录）")
    args = ap.parse_args()
    tab_dir, res_dir = args.out, args.data
    os.makedirs(tab_dir, exist_ok=True)
    with open(os.path.join(res_dir, "results_main.json"), encoding="utf-8") as f:
        res = json.load(f)

    rows = []
    names = ["静态均分", "贪婪调配", "Q-learning", "PPO"]
    for n in names:
        r = res[n]
        rows.append({
            "策略": n,
            "累计物资缺口(箱)": round(r["shortage"]["mean"], 1),
            "缺口std": round(r["shortage"]["std"], 1),
            "累计调运成本": round(r["cost"]["mean"], 1),
            "成本std": round(r["cost"]["std"], 1),
            "物资保障率": round(r["service"]["mean"], 3),
            "保障率std": round(r["service"]["std"], 3),
            "缺货天数": round(r["stockout_days"]["mean"], 1),
        })
    df = pd.DataFrame(rows)
    # 相对改善列
    base_gap = df.loc[df["策略"] == "贪婪调配", "累计物资缺口(箱)"].values[0]
    base_cost = df.loc[df["策略"] == "贪婪调配", "累计调运成本"].values[0]
    df["缺口较贪婪改善%"] = ((base_gap - df["累计物资缺口(箱)"]) / base_gap * 100).round(1)
    df["成本较贪婪改善%"] = ((base_cost - df["累计调运成本"]) / base_cost * 100).round(1)
    df.to_csv(os.path.join(tab_dir, "table1_main.csv"), index=False, encoding="utf-8-sig")
    print(df.to_string(index=False))

    # 敏感性汇总（PPO 相对贪婪的缺口改善随参数变化）
    with open(os.path.join(res_dir, "results_sensitivity.json"), encoding="utf-8") as f:
        sw = json.load(f)
    rows = []
    for dim, label in [("demand", "需求强度倍率"), ("inventory", "库存倍率"), ("transport", "人力成本权重")]:
        for d in sw[dim]:
            rows.append({
                "维度": label, "取值": d["value"],
                "静态缺口": round(d["静态均分"]["shortage"], 1),
                "贪婪缺口": round(d["贪婪调配"]["shortage"], 1),
                "PPO缺口": round(d["PPO"]["shortage"], 1),
                "PPO较贪婪改善%": round((d["贪婪调配"]["shortage"] - d["PPO"]["shortage"]) / d["贪婪调配"]["shortage"] * 100, 1),
                "静态成本": round(d["静态均分"]["cost"], 1),
                "贪婪成本": round(d["贪婪调配"]["cost"], 1),
                "PPO成本": round(d["PPO"]["cost"], 1),
            })
    pd.DataFrame(rows).to_csv(os.path.join(tab_dir, "table2_sensitivity.csv"), index=False, encoding="utf-8-sig")
    print("\n敏感性表已保存")


if __name__ == "__main__":
    main()
