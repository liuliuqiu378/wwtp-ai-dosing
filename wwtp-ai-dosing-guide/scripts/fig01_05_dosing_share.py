"""
图 01-5：典型市政厂药剂费内部结构饼图（合成典型占比，因厂而异）
运行：python3 fig01_05_dosing_share.py
输出：../images/fig01_05_dosing_share.png
说明：占比为行业典型经验量级，随排放标准、工艺与水质差异极大，不代表任何具体水厂。
"""
import matplotlib
matplotlib.use("Agg")
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC"]
plt.rcParams["axes.unicode_minus"] = False

labels = ["外加碳源", "除磷剂 PAC/PFS", "PAM 助凝与脱水", "消毒药剂",
          "其他 酸碱/消泡/营养盐等"]
sizes = [35, 30, 12, 8, 15]
colors = ["#1565c0", "#c62828", "#ef6c00", "#2e7d32", "#8e24aa"]
explode = [0.06, 0.02, 0.0, 0.0, 0.0]

fig, (axl, axr) = plt.subplots(1, 2, figsize=(11.5, 5.3), dpi=130,
                               gridspec_kw={"width_ratios": [1.15, 1.0]})

wedges, texts, autotexts = axl.pie(
    sizes, labels=None, colors=colors, explode=explode,
    autopct=lambda p: f"{p:.0f}%", startangle=90,
    pctdistance=0.72, wedgeprops=dict(edgecolor="white", linewidth=1.5))
for at in autotexts:
    at.set_color("white")
    at.set_fontsize(10)
axl.legend(wedges, labels, loc="lower center", bbox_to_anchor=(0.5, -0.18),
           fontsize=9, ncol=1)
axl.set_title("典型市政厂药剂费内部结构（经验占比）")

# 折算示意：10 万吨/天厂，吨水药剂费取典型中值 0.12 元
cost_per_m3 = 0.12
flow = 100000
annual = cost_per_m3 * flow * 365 / 10000.0   # 万元/年
amounts = [annual * s / 100.0 for s in sizes]
bars = axr.barh(range(len(labels)), amounts, color=colors)
axr.set_yticks(range(len(labels)))
axr.set_yticklabels([l.split()[0] for l in labels], fontsize=9)
axr.invert_yaxis()
axr.set_xlabel("年费（万元/年）")
axr.set_title(f"折算 10 万吨/天厂（吨水药剂费 {cost_per_m3:.2f} 元）")
axr.grid(axis="x", alpha=.3)
for rect, v in zip(bars, amounts):
    axr.text(v + 1.5, rect.get_y() + rect.get_height() / 2,
             f"{v:.0f} 万元", va="center", fontsize=9)
axr.set_xlim(0, max(amounts) * 1.25)

fig.suptitle("药剂费花在了谁身上：碳源与除磷剂合计约占三分之二", fontsize=13)
fig.text(0.5, -0.02,
         "注：占比为合成/典型参数实算，因排放标准、工艺与水质差异极大；"
         "提标到准IV或 TN 压力大的厂，碳源占比可能超过一半。",
         ha="center", fontsize=8.5, color="#555")
plt.tight_layout(rect=(0, 0.03, 1, 1))
plt.savefig("../images/fig01_05_dosing_share.png")
print("已保存 ../images/fig01_05_dosing_share.png")

df = pd.DataFrame({"药剂类别": labels, "占比%": sizes,
                   "年费（万元/年）": [round(a, 1) for a in amounts]})
print(df.to_string(index=False))
print(f"全年药剂费合计约: {annual:.0f} 万元/年（吨水 {cost_per_m3:.2f} 元口径）")
