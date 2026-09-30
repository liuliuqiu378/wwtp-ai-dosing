"""
图 03-2（之二）：不同处理规模 × 排放标准下的吨水药剂费区间（分组柱状区间图）
运行：python3 fig03_02_dosing_intensity.py
输出：../images/fig03_02_dosing_intensity.png
说明：区间为多厂审计经验量级（元/m3），受工艺类型、进水水质、地区价格影响很大；
规模越大、议价与规模效应越强，吨水药耗强度越低；准IV类因深度脱氮除磷显著抬高药耗。
"""
import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC"]
plt.rcParams["axes.unicode_minus"] = False

scales = ["1 万吨/日", "5 万吨/日", "10 万吨/日", "20 万吨/日"]
x = np.arange(len(scales))

# 吨水药剂费区间（元/m3）：（下限，上限）
a1a_low  = [0.10, 0.08, 0.07, 0.06]   # 一级A
a1a_high = [0.25, 0.20, 0.18, 0.15]
iv_low   = [0.18, 0.15, 0.12, 0.10]   # 准IV类
iv_high  = [0.40, 0.32, 0.28, 0.24]

fig, ax = plt.subplots(figsize=(10.5, 6))
w = 0.32

def draw_range(pos, low, high, color, label):
    mid = [(l + h) / 2 for l, h in zip(low, high)]
    err = [[m - l for m, l in zip(mid, low)],
           [h - m for m, h in zip(mid, high)]]
    ax.bar(pos, mid, width=w, color=color, alpha=0.35, label=label, zorder=2)
    ax.errorbar(pos, mid, yerr=err, fmt="o", color=color, lw=2,
                capsize=7, capthick=2, ms=6, zorder=3)
    for p, l, h in zip(pos, low, high):
        ax.text(p, h + 0.012, f"{l:.2f}~{h:.2f}", ha="center",
                fontsize=9, color=color)

draw_range(x - w/2, a1a_low, a1a_high, "#1565c0", "一级A（GB 18918-2002）")
draw_range(x + w/2, iv_low, iv_high, "#c62828", "准IV类（提标口径）")

ax.set_xticks(x)
ax.set_xticklabels(scales, fontsize=11)
ax.set_ylabel("吨水药剂费（元/m3）")
ax.set_ylim(0, 0.48)
ax.set_title("不同规模与排放标准下的吨水药剂费区间（经验量级，具体项目以实测为准）",
             fontsize=12.5)
ax.legend(fontsize=11, frameon=False)
ax.grid(axis="y", alpha=.3)

plt.tight_layout()
out = "../images/fig03_02_dosing_intensity.png"
plt.savefig(out, dpi=130)
print("saved:", out)
for i, s in enumerate(scales):
    print(f"{s}  一级A {a1a_low[i]:.2f}~{a1a_high[i]:.2f} | "
          f"准IV {iv_low[i]:.2f}~{iv_high[i]:.2f} 元/m3")
