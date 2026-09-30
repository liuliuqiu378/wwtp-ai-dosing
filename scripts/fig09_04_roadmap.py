"""
图 09-4：AI加药产品 V1->V2->V3 版本路线时间线（甘特条形）
V1（0-6月）：单药剂（除磷剂）开环建议+台账看板
V2（6-15月）：双药剂半自动闭环+出水软测量
V3（15-30月）：多药剂多变量优化+集团多厂对标
运行：python3 fig09_04_roadmap.py
"""
import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC"]
plt.rcParams["axes.unicode_minus"] = False

bars = [
    ("V1 单药剂开环建议", 0, 6, "#1565c0"),
    ("数据通道与网关", 0, 4, "#90a4ae"),
    ("V2 双药剂半自动闭环", 6, 9, "#ef6c00"),
    ("出水TP/TN软测量", 8, 6, "#f9a825"),
    ("V3 多药剂多变量优化", 15, 10, "#2e7d32"),
    ("集团多厂对标平台", 20, 10, "#6a1b9a"),
]

milestones = [(6, "V1发布"), (15, "V2发布"), (30, "V3发布")]

fig, ax = plt.subplots(figsize=(10.5, 5.6))
for i, (name, start, dur, color) in enumerate(bars):
    y = len(bars) - 1 - i
    ax.barh(y, dur, left=start, height=0.55, color=color, alpha=0.9, edgecolor="white")
    ax.text(start + 0.3, y, name, va="center", ha="left", fontsize=9.3, color="white", fontweight="bold")

for m, label in milestones:
    ax.axvline(m, color="#c62828", ls="--", lw=1.2, alpha=0.7)
    ax.text(m, -0.75, label, ha="center", va="center", fontsize=10, color="#c62828", fontweight="bold")

ax.set_yticks([])
ax.set_xlim(0, 31)
ax.set_ylim(-1.5, len(bars)-0.1)
ax.set_xticks(np.arange(0, 31, 3))
ax.set_xlabel("项目启动后月份", labelpad=32)
ax.set_title("AI加药产品 V1 → V2 → V3 路线图（典型节奏，可按客户基础压缩或拉长）", fontsize=13)
ax.grid(axis="x", alpha=.3)
ax.text(0, -1.35, "数据为产品规划示例；V1先验证价值与信任，V2做闭环与软测量，V3进集团级多厂协同。",
        fontsize=9, color="#666")
plt.tight_layout()
plt.subplots_adjust(bottom=0.18)
plt.savefig("../images/fig09_04_roadmap.png", dpi=130)
plt.close()
print("roadmap figure saved: V1 0-6m, V2 6-15m, V3 15-30m")
