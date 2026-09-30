"""
图 08-1：AI 加药项目八阶段实施甘特图（典型六到八个月口径，本例排到 30 周约 7 个月）
运行：python3 fig08_01_gantt.py
输出：../images/fig08_01_gantt.png
说明：周数为典型项目经验值，阶段四/五与阶段六/七允许搭接；
具体工期随改造量、化验数据可得性、停水窗口与验收严格程度浮动。
"""
import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC"]
plt.rcParams["axes.unicode_minus"] = False

# 阶段：名称，起始周，持续周数（典型取值），颜色
phases = [
    ("①商务调研与工艺审计", 0, 2, "#90caf9"),
    ("②数据审计与基线建立", 2, 4, "#64b5f6"),
    ("③方案设计与选型", 6, 2, "#42a5f5"),
    ("④点位网络加药设备改造", 8, 5, "#ffb74d"),
    ("⑤边缘软硬件部署与打通", 12, 3, "#ffd54f"),
    ("⑥模型开发加影子运行", 14, 6, "#81c784"),
    ("⑦灰度投运 建议半自动限定自动", 20, 6, "#a5d6a7"),
    ("⑧验收与持续运营启动", 26, 2, "#ce93d8"),
]
# 关键里程碑：周次，标签
milestones = [(2, "M1 审计报告"), (6, "M2 方案冻结"),
              (14, "M3 数据打通"), (20, "M4 影子达标"),
              (26, "M5 自动投运"), (28, "M6 验收")]

fig, ax = plt.subplots(figsize=(11.5, 6.2))
y = np.arange(len(phases))[::-1]
for (name, start, dur, color), yy in zip(phases, y):
    ax.broken_barh([(start, dur)], (yy - 0.38, 0.76),
                   facecolors=color, edgecolors="#455a64", linewidth=1.0, zorder=3)
    ax.text(start + dur / 2, yy, f"{dur} 周", ha="center", va="center",
            fontsize=10, color="#263238", zorder=4)

import matplotlib.transforms as mtrans
_xform = mtrans.blended_transform_factory(ax.transData, ax.transAxes)
for i, (week, label) in enumerate(milestones):
    ax.axvline(week, color="#c62828", ls="--", lw=1.1, alpha=0.55, zorder=2)
    ylab = 0.965 if i % 2 == 0 else 0.905
    ax.text(week, ylab, label, transform=_xform, rotation=0, ha="center",
            va="center", fontsize=8.5, color="#b71c1c",
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75))

ax.set_yticks(y)
ax.set_yticklabels([p[0] for p in phases], fontsize=11)
ax.set_xlim(0, 30)
ax.set_xticks(range(0, 31, 2))
ax.set_xticklabels([f"第{w}周" for w in range(0, 31, 2)], fontsize=9)
ax.set_xlabel("项目周（含阶段搭接，本例排期 28 周约 6.5 个月，落在六到八个月典型区间；顺利可压到 6 个月，改造量大可延至 8 个月以上）")
ax.set_title("AI 加药管控项目八阶段实施甘特图（典型经验排期，具体项目以实际合同为准）", fontsize=12.5)
ax.grid(axis="x", alpha=.3, zorder=0)

plt.tight_layout()
out = "../images/fig08_01_gantt.png"
plt.savefig(out, dpi=130)
print("saved:", out)
total_critical = phases[-1][1] + phases[-1][2]
print("总工期周数(搭接排期):", total_critical, "约", round(total_critical / 4.33, 1), "个月")
for name, s, d, _ in phases:
    print(f"{name}: 第{s}-{s+d}周")
