"""
图 09-3：AI加药 TAM-SAM-SOM 市场漏斗（存量一次性建设口径，示例测算）
- TAM：约3000座 1万吨/日以上城镇厂及可比工业园区设施 × 单厂均价80万 = 24亿
- SAM：仪表与自控基础达标、近三年有提标/智慧化任务的约1500座 × 80万 = 12亿
- SOM：未来三年累计渗透15% 约225座，建设费+订阅三年合同额约1.71亿
所有数字为方法论演示用示例假设，非市场预测；厂数基数以住建部《城市建设统计年鉴》最新版为准。
运行：python3 fig09_03_market_funnel.py
"""
import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC"]
plt.rcParams["axes.unicode_minus"] = False

stages = [
    ("TAM 总潜在市场", "约3000座 × 80万/座", 24.00, 10.0, 7.6, "#1565c0"),
    ("SAM 可服务市场", "约1500座 × 80万/座", 12.00, 7.2, 4.8, "#00838f"),
    ("SOM 三年可获取", "累计渗透15% 约225座（含订阅）", 1.71, 4.4, 2.2, "#c62828"),
]

fig, ax = plt.subplots(figsize=(11, 6.4))
cx = 4.6
band_h = 0.22
gap = 0.115
y_top0 = 0.97

for i, (title, sub, val, w_top, w_bot, color) in enumerate(stages):
    y_top = y_top0 - i * (band_h + gap)
    y_bot = y_top - band_h
    poly = Polygon([(cx - w_top / 2, y_top), (cx + w_top / 2, y_top),
                    (cx + w_bot / 2, y_bot), (cx - w_bot / 2, y_bot)],
                   closed=True, facecolor=color, edgecolor="white", lw=2, alpha=0.92)
    ax.add_patch(poly)
    ax.text(cx, y_top - 0.075, title, ha="center", va="center",
            fontsize=13, color="white", fontweight="bold")
    ax.text(cx, y_bot + 0.055, f"{val:.2f} 亿元", ha="center", va="center",
            fontsize=15, color="white", fontweight="bold")
    ax.text(cx, y_bot - gap / 2, sub, ha="center", va="center", fontsize=10, color="#444")

years = ["第1年（渗透3%，45座）", "第2年（累计8%，120座）", "第3年（累计15%，225座）"]
new_units = [45, 75, 105]
recur = [0, 360, 960]
new_rev = [u * 70 for u in new_units]
for j, (yr, nr, rc) in enumerate(zip(years, new_rev, recur)):
    yc = 0.86 - j * 0.27
    txt = f"{yr}\n当年新签 {nr} 万\n存量续费 {rc} 万"
    ax.text(9.4, yc, txt, ha="center", va="center", fontsize=9.5,
            bbox=dict(boxstyle="round,pad=0.5", fc="#fff3e0", ec="#ef6c00", lw=1.3))

ax.set_xlim(0, 12.6)
ax.set_ylim(-0.13, 1.05)
ax.axis("off")
ax.set_title("AI加药市场测算漏斗（存量建设口径，示例假设，非市场预测）", fontsize=15, pad=12)
ax.text(0.2, -0.105, "数据为方法论演示用合成参数；行业基数以住建部《城市建设统计年鉴》最新版为准，落地前以自下而上区域复核为准。",
        fontsize=9, color="#666")
plt.tight_layout()
plt.savefig("../images/fig09_03_market_funnel.png", dpi=130)
plt.close()

cum_units = np.cumsum(new_units)
print("年份 新增座次 累计座次 新签收入(万) 订阅收入(万) 当年合同额(亿)")
for yr, u, cum, nr, rc in zip(["第1年", "第2年", "第3年"], new_units, cum_units, new_rev, recur):
    print(f"{yr} {u:4d} {cum:5d} {nr:8d} {rc:8d} {(nr+rc)/10000:8.2f}")
print(f"三年累计合同额: {sum(new_rev)+sum(recur)} 万元 = {(sum(new_rev)+sum(recur))/10000:.2f} 亿元")
print("漏斗: TAM 24亿 / SAM 12亿 / 三年SOM %.2f亿（含订阅）" % ((sum(new_rev)+sum(recur))/10000))
