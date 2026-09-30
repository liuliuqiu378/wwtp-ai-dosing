"""
图 06-1：AI 加药五类能力地图
横轴：落地成熟度（1=观望，3=试点，5=成熟）
纵轴：10 万吨级厂的典型年收益量级（1=小，5=很大）
气泡大小：数据门槛（越大代表对数据量与质量要求越高）
运行：python3 fig06_01_capability_map.py
输出：../images/fig06_01_capability_map.png
说明：评分为行业项目经验的定性估计（合成/典型口径），具体项目以实测为准。
"""
import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC"]
plt.rcParams["axes.unicode_minus"] = False

rng = np.random.default_rng(42)

# 五类能力：编号 名称 成熟度 收益量级 数据门槛（1低~5高）
caps = [
    ("①", "出水指标预测与预警", 4.2, 3.8, 3.0),
    ("②", "加药量前馈推荐",     4.0, 4.6, 3.0),
    ("③", "多目标优化决策",     2.6, 4.3, 4.5),
    ("④", "异常检测与工况识别", 3.6, 3.3, 2.5),
    ("⑤", "机器视觉",           2.1, 3.0, 4.0),
]
df = pd.DataFrame(caps, columns=["编号", "能力", "成熟度", "收益", "数据门槛"])
print(df.to_string(index=False))

colors = {"①": "#1565c0", "②": "#2e7d32", "③": "#ef6c00",
          "④": "#6a1b9a", "⑤": "#00838f"}

fig, ax = plt.subplots(figsize=(11, 7), dpi=130)

# 背景分区：观望 / 试点 / 成熟
ax.axvspan(0.5, 2.5, color="#eceff1", alpha=.8)
ax.axvspan(2.5, 3.5, color="#fff8e1", alpha=.8)
ax.axvspan(3.5, 5.0, color="#e8f5e9", alpha=.8)
ax.text(1.5, 2.35, "观望区", ha="center", fontsize=11, color="#607d8b")
ax.text(3.0, 2.35, "试点区", ha="center", fontsize=11, color="#b28704")
ax.text(4.35, 2.35, "成熟可复制区", ha="center", fontsize=11, color="#2e7d32")

for _, r in df.iterrows():
    ax.scatter(r["成熟度"], r["收益"],
               s=600 + r["数据门槛"] * 520,
               color=colors[r["编号"]], alpha=.55,
               edgecolor="white", linewidth=1.5, zorder=3)
    ax.text(r["成熟度"], r["收益"], r["编号"], ha="center", va="center",
            fontsize=13, fontweight="bold", color="white", zorder=4)
    ax.annotate(r["能力"], xy=(r["成熟度"], r["收益"]),
                xytext=(r["成熟度"] + 0.12, r["收益"] + 0.22),
                fontsize=10.5, fontweight="bold", color="#263238")

ax.set_xlim(0.5, 5.1)
ax.set_ylim(2.2, 5.1)
ax.set_xticks([1, 2, 3, 4, 5])
ax.set_xticklabels(["1\n观望", "2", "3\n试点", "4", "5\n成熟"])
ax.set_yticks([2, 3, 4, 5])
ax.set_xlabel("落地成熟度（项目可复制性）", fontsize=11.5)
ax.set_ylabel("10 万吨级厂典型年收益量级", fontsize=11.5)
ax.set_title("AI 加药五类能力地图：越靠右越成熟，越靠上越值钱，气泡越大越吃数据",
             fontsize=12.5, fontweight="bold")
ax.grid(alpha=.3)

legend_size = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor="#90a4ae",
           markersize=11, label="数据门槛低"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor="#90a4ae",
           markersize=17, label="数据门槛中"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor="#90a4ae",
           markersize=23, label="数据门槛高"),
]
ax.legend(handles=legend_size, loc="lower right", fontsize=9.5, framealpha=.9)

plt.tight_layout()
out = "../images/fig06_01_capability_map.png"
plt.savefig(out)
print("saved:", out)
