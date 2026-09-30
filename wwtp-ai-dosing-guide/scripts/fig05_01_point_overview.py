"""
图 05-1：典型加药 AI 项目的点位全景
左：八个分区的点位数量（与正文点位表一致，共 42 点）
右：三类数据源的采集周期 vs 每日入库条数（双对数，气泡大小=变量数）
运行：python3 fig05_01_point_overview.py
输出：../images/fig05_01_point_overview.png
说明：本图为概念性实算演示，参数为典型经验量级，不代表任何具体水厂。
"""
import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC"]
plt.rcParams["axes.unicode_minus"] = False

rng = np.random.default_rng(42)

# ---- 左图：分区点数（与正文点位表逐行对应，合计 42）----
zones = ["进水", "生物处理", "曝气", "加药", "二沉深度", "污泥", "出水", "能耗"]
counts = [7, 4, 4, 8, 4, 4, 7, 4]
colors = ["#1565c0", "#2e7d32", "#00838f", "#c62828",
          "#6a1b9a", "#4e342e", "#ad1457", "#f9a825"]

# ---- 右图：三类数据源 ----
# PLC/SCADA：42 点按 1 min 归档；LIMS：10 个指标、一天 3 次；业务：8 类台账一天 1 批
sources = pd.DataFrame({
    "name": ["PLC/SCADA 实时过程数据", "化验室 LIMS 化验数据", "业务管理数据"],
    "period_min": [1.0, 480.0, 1440.0],
    "vars": [42, 10, 8],
})
sources["records_day"] = sources["vars"] * (1440.0 / sources["period_min"])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 5.6), dpi=130)

# ---- 左：横向条形 ----
y = np.arange(len(zones))
bars = ax1.barh(y, counts, color=colors, alpha=.85, edgecolor="white")
ax1.set_yticks(y)
ax1.set_yticklabels(zones, fontsize=11)
ax1.invert_yaxis()
ax1.set_xlabel("接入点位数量（个）", fontsize=11)
ax1.set_title("典型加药AI项目点位构成：共42点", fontsize=13, fontweight="bold")
for b, c in zip(bars, counts):
    ax1.text(b.get_width() + 0.15, b.get_y() + b.get_height()/2,
             f"{c}点", va="center", fontsize=10)
ax1.set_xlim(0, 9.5)
ax1.grid(axis="x", alpha=.3)

# ---- 右：周期 vs 日入库条数 ----
markers = ["o", "s", "^"]
src_colors = ["#1565c0", "#2e7d32", "#f9a825"]
for i, row in sources.iterrows():
    ax2.scatter(row["period_min"], row["records_day"],
                s=180 + row["vars"]*55, color=src_colors[i],
                marker=markers[i], alpha=.75, edgecolor="k", linewidths=.8,
                zorder=3)
    ax2.annotate(f'{row["name"]}\n{row["vars"]}个变量，'
                 f'{row["records_day"]:.0f}条/日',
                 xy=(row["period_min"], row["records_day"]),
                 xytext=(8, 14 if i == 0 else -26), textcoords="offset points",
                 fontsize=9.5)
ax2.set_xscale("log")
ax2.set_yscale("log")
ax2.set_xlabel("典型采集/归档周期（分钟，对数轴）", fontsize=11)
ax2.set_ylabel("每日入库记录条数（条，对数轴）", fontsize=11)
ax2.set_title("三类数据源：周期差约3个数量级，条数差约4个数量级",
              fontsize=12.5, fontweight="bold")
ax2.set_xlim(0.3, 5000)
ax2.set_ylim(2, 200000)
ax2.grid(alpha=.3, which="both")


plt.tight_layout()
out = "../images/fig05_01_point_overview.png"
plt.savefig(out)
print("saved:", out)
print("点位合计:", sum(counts))
print(sources.to_string(index=False))
