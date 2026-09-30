"""
图 01-3：各处理段沿程浓度变化（初沉 → 生物 → 深度），合成典型市政厂算例
运行：python3 fig01_03_removal_stage.py
输出：../images/fig01_03_removal_stage.png
说明：本图为典型设计量级演示，不代表任何具体水厂；实际沿程浓度以现场实测为准。
"""
import matplotlib
matplotlib.use("Agg")
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC"]
plt.rcParams["axes.unicode_minus"] = False

stages = ["进厂进水", "格栅+沉砂", "初沉出水", "生物+二沉", "深度出水", "消毒出水"]

# 沿程浓度矩阵（mg/L），SS 为典型设计示例
data = pd.DataFrame({
    "COD":  [350.0, 330.0, 250.0, 45.0, 25.0, 25.0],
    "SS":   [220.0, 200.0, 120.0, 15.0,  6.0,  6.0],
    "氨氮": [35.0,  35.0,  34.0,  2.0,  1.2,  1.0],
    "TN":   [45.0,  45.0,  42.0, 13.0, 10.0, 10.0],
    "TP":   [4.5,   4.3,   3.5,   0.9,  0.25, 0.25],
}, index=stages)

x = np.arange(len(stages))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 5.3), dpi=130)

for col, color in [("COD", "#6d4c41"), ("SS", "#546e7a")]:
    ax1.plot(x, data[col], marker="o", lw=2.2, color=color, label=col)
    for xi, yi in zip(x, data[col]):
        ax1.text(xi, yi + 8, f"{yi:g}", ha="center", fontsize=8, color=color)
ax1.set_xticks(x)
ax1.set_xticklabels(stages, rotation=20, ha="right", fontsize=8.5)
ax1.set_ylabel("浓度 (mg/L)")
ax1.set_title("碳污染与悬浮物：主要靠 初沉 + 生物 两段吃掉")
ax1.set_ylim(0, 400)
ax1.grid(alpha=.3)
ax1.legend(fontsize=9)

for col, color in [("氨氮", "#2e7d32"), ("TN", "#1565c0"), ("TP", "#c62828")]:
    ax2.plot(x, data[col], marker="s", lw=2.2, color=color, label=col)
    for xi, yi in zip(x, data[col]):
        ax2.text(xi, yi + 1.1, f"{yi:g}", ha="center", fontsize=8, color=color)
ax2.set_xticks(x)
ax2.set_xticklabels(stages, rotation=20, ha="right", fontsize=8.5)
ax2.set_ylabel("浓度 (mg/L)")
ax2.set_title("氮与磷：生物段脱氮，深度段化学除磷兜底")
ax2.axhline(0.5, color="#c62828", ls=":", lw=1.2)
ax2.text(0.1, 0.55, "一级A TP 0.5", color="#c62828", fontsize=8)
ax2.set_ylim(0, 50)
ax2.grid(alpha=.3)
ax2.legend(fontsize=9)

fig.suptitle("污水在各工段经历了什么：五项指标沿程浓度阶梯（合成典型算例）", fontsize=13)
fig.text(0.5, -0.02, "数据为合成/典型参数实算：消毒主要杀菌，对常规浓度指标影响很小。",
         ha="center", fontsize=8.5, color="#555")
plt.tight_layout(rect=(0, 0.03, 1, 1))
plt.savefig("../images/fig01_03_removal_stage.png")
print("已保存 ../images/fig01_03_removal_stage.png")

removal = (1 - data.iloc[-1] / data.iloc[0]) * 100
print("全厂总去除率（进厂进水 → 消毒出水）:")
for k, v in removal.items():
    print(f"  {k}: {v:.1f}%")
print("分级去除贡献（占该指标进厂总量）:")
for k in ["COD", "SS", "氨氮", "TN", "TP"]:
    s = data[k]
    p_prim = (s.iloc[0] - s.iloc[2]) / s.iloc[0] * 100
    p_bio = (s.iloc[2] - s.iloc[3]) / s.iloc[0] * 100
    p_adv = (s.iloc[3] - s.iloc[5]) / s.iloc[0] * 100
    print(f"  {k}: 初沉 {p_prim:.1f}% | 生物段 {p_bio:.1f}% | 深度段 {p_adv:.1f}%")
