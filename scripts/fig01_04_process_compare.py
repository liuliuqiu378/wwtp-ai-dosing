"""
图 01-4：AAO / 氧化沟 / SBR-CASS / MBR 六维度相对评分雷达图（1-5 分经验评价）
运行：python3 fig01_04_process_compare.py
输出：../images/fig01_04_process_compare.png
说明：分数为工程经验相对评价，非实测；5 分代表该维度表现最好，仅供选型直觉建立。
"""
import matplotlib
matplotlib.use("Agg")
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC"]
plt.rcParams["axes.unicode_minus"] = False

dims = ["占地紧凑", "节能性", "投资经济性", "抗冲击能力", "出水保障", "运维简便"]
scores = pd.DataFrame({
    "AAO":      [3, 4, 4, 3, 3, 4],
    "氧化沟":    [2, 3, 4, 5, 3, 4],
    "SBR/CASS": [3, 3, 3, 4, 3, 2],
    "MBR":      [5, 1, 1, 3, 5, 2],
}, index=dims)

angles = np.linspace(0, 2 * np.pi, len(dims), endpoint=False).tolist()
angles_closed = angles + angles[:1]

fig, ax = plt.subplots(figsize=(8.5, 7), dpi=130, subplot_kw=dict(polar=True))
colors = {"AAO": "#1565c0", "氧化沟": "#2e7d32", "SBR/CASS": "#ef6c00", "MBR": "#6a1b9a"}
markers = {"AAO": "o", "氧化沟": "s", "SBR/CASS": "^", "MBR": "D"}

for name in scores.columns:
    vals = scores[name].tolist()
    vals_closed = vals + vals[:1]
    ax.plot(angles_closed, vals_closed, lw=2, color=colors[name],
            marker=markers[name], ms=5, label=name)
    ax.fill(angles_closed, vals_closed, color=colors[name], alpha=.08)

ax.set_xticks(angles)
ax.set_xticklabels(dims, fontsize=11)
ax.set_ylim(0, 5)
ax.set_yticks([1, 2, 3, 4, 5])
ax.set_yticklabels(["1", "2", "3", "4", "5"], fontsize=8, color="#888")
ax.set_title("四大主流生物工艺六维对比（1-5 分，5 分最好，经验评价）", pad=22, fontsize=13)
ax.legend(loc="upper right", bbox_to_anchor=(1.28, 1.08), fontsize=10)
fig.text(0.5, 0.02,
         "注：评分为相对量级的工程经验判断（如 MBR 出水最好但最费电、膜更换贵；氧化沟最抗冲击但占地大），"
         "具体项目以水质水量与占地条件为准。",
         ha="center", fontsize=8.5, color="#555")
plt.tight_layout(rect=(0, 0.04, 1, 1))
plt.savefig("../images/fig01_04_process_compare.png")
print("已保存 ../images/fig01_04_process_compare.png")
print(scores)
print("\n各工艺总分（满分30）:")
print(scores.sum().sort_values(ascending=False).to_string())
