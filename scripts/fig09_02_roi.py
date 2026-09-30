"""
图 09-2（两图）：
1) fig09_02_cashflow.png  2万/10万/20万吨三档厂累计净现金流回收曲线（36个月）
2) fig09_02_tornado.png   10万吨厂回收期对各因素的敏感性龙卷风图

口径（典型经验值，具体项目以实测为准）：
- 投资发生在第0月（含硬件、软件、实施集成、首年服务）
- 年节约为四本账合计的中性测算（毛口径，未扣年运维；扣运维口径见正文）
- 2万吨厂：投资45万，年节约约25万；10万吨厂：投资100万，年节约约77万；20万吨厂：投资150万，年节约约143万
运行：python3 fig09_02_roi.py
"""
import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC"]
plt.rcParams["axes.unicode_minus"] = False

# ---------------- 三档厂参数 ----------------
tiers = [
    {"name": "2万吨厂", "invest": 45.0, "save": 25.4, "color": "#1565c0"},
    {"name": "10万吨厂", "invest": 100.0, "save": 77.0, "color": "#c62828"},
    {"name": "20万吨厂", "invest": 150.0, "save": 142.8, "color": "#2e7d32"},
]

months = np.arange(0, 37)

fig, ax = plt.subplots(figsize=(10.5, 6.2))
for t in tiers:
    monthly = t["save"] / 12.0
    cash = -t["invest"] + monthly * months
    ax.plot(months, cash, lw=2.4, color=t["color"], label=t["name"])
    # 回收期交点
    pb = t["invest"] / monthly
    ax.scatter([pb], [0], color=t["color"], zorder=5, s=55)
    ax.annotate(f"{pb:.1f}个月", xy=(pb, 0), xytext=(pb + 0.6, 8 - tiers.index(t) * 7),
                fontsize=10, color=t["color"], fontweight="bold")
    # 第36个月累计收益
    ax.annotate(f"{cash[-1]:.0f}万", xy=(36, cash[-1]), xytext=(36.3, cash[-1]),
                fontsize=9, color=t["color"], va="center")

ax.axhline(0, color="#555", lw=1.2)
ax.set_xlabel("投运后月份")
ax.set_ylabel("累计净现金流（万元）")
ax.set_title("三档厂累计净现金流与静态回收期（投资于第0月一次投入）", fontsize=13)
ax.set_xticks(np.arange(0, 37, 3))
ax.grid(alpha=.3)
ax.legend(loc="upper left", frameon=False)
ax.text(0.5, -118, "数据为典型参数实算（具体项目以实测为准）；年节约未扣年运维费，扣运维口径见正文。",
        fontsize=9, color="#666")
plt.tight_layout()
plt.savefig("../images/fig09_02_cashflow.png", dpi=130)
plt.close()

print("=== 回收期（毛口径：投资/月节约） ===")
for t in tiers:
    pb = t["invest"] / (t["save"] / 12.0)
    print(f"{t['name']}: 投资{t['invest']:.0f}万 年节约{t['save']:.1f}万 回收期{pb:.1f}个月 36个月累计{t['save']*3-t['invest']:.1f}万")

# ---------------- 龙卷风图：10万吨厂 ----------------
invest0 = 100.0
save0 = 77.0          # 中性年节约
chem0 = 69.2          # 其中药剂类（除磷35.7+碳源33.5），随节药率/药价变动
pb0 = invest0 / save0 * 12

def pb(saving, invest=invest0):
    return invest / saving * 12

# 各因素的（低回收期，高回收期）
factors = []

# 1 两种药剂综合节药率同时 +/- 5个百分点（极端组合压力测试）
d_rate = 223.4 * 0.05 + 372.3 * 0.05
factors.append(("综合节药率 ±5个百分点", pb(save0 + d_rate), pb(save0 - d_rate)))

# 2 系统投资 +/- 20%
factors.append(("系统投资 ±20%", pb(save0, invest0 * 0.8), pb(save0, invest0 * 1.2)))

# 3 药剂单价 +/- 20%
d_price = chem0 * 0.20
factors.append(("药剂单价 ±20%", pb(save0 + d_price), pb(save0 - d_price)))

# 4 进水负荷率 +/- 15%（节约随水量近似线性）
factors.append(("进水负荷率 ±15%", pb(save0 * 1.15), pb(save0 * 0.85)))

# 5 年运维/云服务费 0~10万（从年节约中扣减）
factors.append(("年运维费 0~10万", pb(save0 - 0), pb(save0 - 10)))

# 按摆动幅度排序
factors.sort(key=lambda f: f[2] - f[1], reverse=True)

fig2, ax2 = plt.subplots(figsize=(10.5, 5.6))
y = np.arange(len(factors))[::-1]
for yi, (name, lo, hi) in zip(y, factors):
    ax2.barh(yi, lo - pb0, left=pb0, color="#2e7d32", height=0.55)
    ax2.barh(yi, hi - pb0, left=pb0, color="#c62828", height=0.55)
    ax2.text(lo - 0.3, yi, f"{lo:.1f}", ha="right", va="center", fontsize=9, color="#2e7d32")
    ax2.text(hi + 0.3, yi, f"{hi:.1f}", ha="left", va="center", fontsize=9, color="#c62828")
    ax2.text(10.3, yi, name, ha="right", va="center", fontsize=10, clip_on=False)

ax2.axvline(pb0, color="#333", lw=1.4, ls="--")
ax2.text(pb0, 4.45, f"基准 {pb0:.1f}个月", ha="center", va="center", fontsize=10, fontweight="bold",
         bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#333"))
ax2.set_xlabel("静态回收期（月）  绿=有利情形  红=不利情形", labelpad=34)
ax2.set_title("回收期敏感性龙卷风图（10万吨厂，基准：投资100万、年节约77万）", fontsize=13)
ax2.set_xlim(10.5, 30)
ax2.set_ylim(-1.0, 5.0)
ax2.set_xticks(np.arange(10, 31, 2.5))
ax2.set_yticks([])
ax2.grid(axis="x", alpha=.3)
ax2.text(10.7, -1.45, "数据为典型参数实算（具体项目以实测为准）；节药率为决定性变量，故合同只承诺区间下限。",
         fontsize=9, color="#666")
plt.tight_layout()
plt.subplots_adjust(bottom=0.16)
plt.savefig("../images/fig09_02_tornado.png", dpi=130)
plt.close()

print("\n=== 龙卷风：基准回收期 %.1f 个月 ===" % pb0)
for name, lo, hi in factors:
    print(f"{name}: {lo:.1f} ~ {hi:.1f} 个月（摆幅 {hi-lo:.1f}）")

# NPV 算例（折现率8%，3年，年净收益=年节约-年运维8万）
r = 0.08
for label, net in [("保守", 70 - 10), ("中性", 77 - 8)]:
    npv = -100 + sum(net / (1 + r) ** k for k in (1, 2, 3))
    print(f"NPV {label}: 年净收益{net:.0f}万 -> 3年NPV={npv:.1f}万；简单ROI={net/100*100:.0f}%")
