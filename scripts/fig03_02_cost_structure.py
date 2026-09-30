"""
图 03-2：成本结构两张饼图
左：市政厂吨水运行成本结构（典型值，合计 0.81 元/m3，落在行业常见 0.6~1.2 元/m3 区间内）
右：药剂费内部结构（典型值，按带深度脱氮、一级A口径厂；无外加碳源厂结构差异极大）
运行：python3 fig03_02_cost_structure.py
输出：../images/fig03_02_cost_structure.png
说明：数字为规范口径区间内选取的典型值，用于建立直觉，不代表任何具体水厂。
"""
import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC"]
plt.rcParams["axes.unicode_minus"] = False

# ---- 左图：吨水运行成本（元/m3，典型值）----
cost_items = ["电费", "药剂费", "污泥处置费", "人工费", "维修费", "折旧及财务"]
cost_val = [0.24, 0.12, 0.14, 0.08, 0.05, 0.18]
cost_colors = ["#1565c0", "#c62828", "#6d4c41", "#2e7d32", "#f9a825", "#546e7a"]

# ---- 右图：药剂费内部结构（%，典型：一级A、需外加碳源脱氮的厂）----
chem_items = ["碳源（乙酸钠等）", "除磷剂（PAC/PFS）", "PAM（污泥脱水）",
              "消毒（次氯酸钠）", "其他（助滤/中和等）"]
chem_pct = [50, 24, 8, 12, 6]
chem_colors = ["#c62828", "#ef6c00", "#f9a825", "#00838f", "#9e9e9e"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 6))

wedges1, _, autotexts1 = ax1.pie(
    cost_val, colors=cost_colors, autopct=lambda p: f"{p:.0f}%",
    startangle=90, pctdistance=0.75,
    wedgeprops=dict(width=0.55, edgecolor="white"))
for t in autotexts1:
    t.set_color("white"); t.set_fontsize(10); t.set_fontweight("bold")
ax1.legend(wedges1,
           [f"{n}  {v:.2f} 元/m3" for n, v in zip(cost_items, cost_val)],
           loc="center left", bbox_to_anchor=(0.92, 0.5), fontsize=10, frameon=False)
ax1.set_title("吨水运行成本结构（典型合计 0.81 元/m3）", fontsize=12)
ax1.text(0, 0, f"合计\n{sum(cost_val):.2f}\n元/m3",
         ha="center", va="center", fontsize=12, fontweight="bold", color="#333")

wedges2, _, autotexts2 = ax2.pie(
    chem_pct, colors=chem_colors, autopct=lambda p: f"{p:.0f}%",
    startangle=90, pctdistance=0.75,
    wedgeprops=dict(width=0.55, edgecolor="white"))
for t in autotexts2:
    t.set_color("white"); t.set_fontsize(10); t.set_fontweight("bold")
# 折算成吨水金额（药剂费按 0.12 元/m3 典型值）
chem_money = [0.12 * p / 100 for p in chem_pct]
ax2.legend(wedges2,
           [f"{n}  {p}%（约{m:.3f} 元/m3）" for n, p, m in zip(chem_items, chem_pct, chem_money)],
           loc="center left", bbox_to_anchor=(0.92, 0.5), fontsize=9.5, frameon=False)
ax2.set_title("药剂费内部结构（一级A、外加碳源厂典型口径）", fontsize=12)
ax2.text(0, 0, "药剂费\n约 0.12\n元/m3",
         ha="center", va="center", fontsize=12, fontweight="bold", color="#333")

plt.tight_layout()
out = "../images/fig03_02_cost_structure.png"
plt.savefig(out, dpi=130, bbox_inches="tight")
print("saved:", out)
print(f"吨水运行成本合计: {sum(cost_val):.2f} 元/m3")
print("药剂费内部结构（元/m3）:",
      {n: round(m, 3) for n, m in zip(chem_items, chem_money)})
