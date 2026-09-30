"""
图 03-3：八大痛点在 10 万吨/日厂对应的年损失量级（横向区间条形图）
运行：python3 fig03_03_pain_loss.py
输出：../images/fig03_03_pain_loss.png
说明：损失区间为多厂审计经验量级（万元/年，含药费、污泥处置、电耗、罚款风险等综合口径）；
八项之间存在重叠，不可直接相加；具体数值高度依赖现场，仅用于建立量级直觉。
"""
import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC"]
plt.rcParams["axes.unicode_minus"] = False

pains = [
    "①进水波动、调节滞后半拍",
    "②过量投加、冲击后不回调",
    "③在线仪表失真致误判",
    "④夜班疲劳、经验不可复制",
    "⑤多药剂各管一段、不协同",
    "⑥多加药→多产泥连锁无人算总账",
    "⑦超标高压下不敢优化不敢试",
    "⑧数据孤岛、经验流失、不可对标",
]
low  = [30, 40, 20, 30, 50, 40, 60, 20]
high = [80, 120, 80, 90, 150, 130, 200, 60]

y = np.arange(len(pains))[::-1]
mid = [(l + h) / 2 for l, h in zip(low, high)]

fig, ax = plt.subplots(figsize=(11, 6.6))
for yi, l, h in zip(y, low, high):
    ax.barh(yi, h - l, left=l, height=0.52, color="#c62828", alpha=0.28, zorder=2)
    ax.plot([l, h], [yi, yi], color="#c62828", lw=2.4, zorder=3)
    ax.plot(mid[list(y).index(yi)], yi, "o", color="#b71c1c", ms=7, zorder=4)
    ax.text(h + 4, yi, f"{l}~{h} 万元", va="center", fontsize=10.5, color="#b71c1c")

ax.set_yticks(y)
ax.set_yticklabels(pains, fontsize=11)
ax.set_xlabel("年损失量级（万元/年，10 万吨/日厂，综合口径）")
ax.set_xlim(0, 245)
ax.set_title("加药管控八大痛点的年损失量级（区间为经验估算，高度依赖现场，不可叠加）",
             fontsize=12.5)
ax.grid(axis="x", alpha=.3)

plt.tight_layout()
out = "../images/fig03_03_pain_loss.png"
plt.savefig(out, dpi=130)
print("saved:", out)
total_mid = sum(mid)
print("八项中值之和（仅示意，实际不可叠加）:", total_mid, "万元/年")
for p, l, h in zip(pains, low, high):
    print(f"{p}: {l}~{h} 万元/年")
