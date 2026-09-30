"""
图 05-2：脏数据的九种死法——3x3 合成时序网格
每个子图演示一种典型脏数据形态，红圈/红底标注问题区段。
运行：python3 fig05_02_dirty_grid.py
输出：../images/fig05_02_dirty_grid.png
说明：本图全部为合成数据，仅用于演示形态，不代表任何具体水厂。
"""
import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC"]
plt.rcParams["axes.unicode_minus"] = False

rng = np.random.default_rng(42)
RED = "#d32f2f"
BLUE = "#1565c0"
t = np.arange(0, 240, dtype=float)          # 240 个分钟点
base = 2.0 + 0.5*np.sin(t/40.0) + rng.normal(0, 0.04, t.size)

fig, axes = plt.subplots(3, 3, figsize=(12, 8.2), dpi=130)
axes = axes.ravel()

def finish(ax, title):
    ax.set_title(title, fontsize=11.5, fontweight="bold")
    ax.tick_params(labelsize=8)
    ax.grid(alpha=.3)

# ① 通讯中断缺失
x = base.copy(); x[95:130] = np.nan
ax = axes[0]
ax.plot(t, x, color=BLUE, lw=1.4)
ax.axvspan(95, 130, color=RED, alpha=.12)
ax.annotate("通讯中断35分钟\n整段无数据", xy=(112, 2.3), ha="center",
            color=RED, fontsize=9, fontweight="bold")
finish(ax, "①通讯中断：整段缺失")

# ② 传感器卡死恒值
x = base.copy(); x[110:165] = 2.18
ax = axes[1]
ax.plot(t, x, color=BLUE, lw=1.4)
ax.scatter([110, 164], [2.18, 2.18], s=90, facecolor="none",
           edgecolor=RED, lw=2, zorder=5)
ax.axhline(2.18, color=RED, ls="--", lw=1, alpha=.6)
ax.annotate("连续55个点完全相同\n方差=0，物理上不可能", xy=(137, 1.55),
            ha="center", color=RED, fontsize=9, fontweight="bold")
finish(ax, "②传感器卡死：恒值一条线")

# ③ 毛刺跳变
x = base.copy()
spikes = [60, 120, 121, 190]
x[spikes] += rng.choice([-1, 1], len(spikes)) * np.array([3.2, 4.5, 4.2, 3.8])
ax = axes[2]
ax.plot(t, x, color=BLUE, lw=1.3)
ax.scatter(t[spikes], x[spikes], s=80, facecolor="none",
           edgecolor=RED, lw=1.8, zorder=5)
ax.annotate("单点突变3~5倍\n下一拍立刻恢复", xy=(120, 4.7),
            ha="center", color=RED, fontsize=9, fontweight="bold")
finish(ax, "③毛刺跳变：单点抽风")

# ④ 量程漂移（探头老化/污染，锯齿式越漂越高，清洗后回落）
x = base.copy()
x += 0.006*t                                    # 缓漂
for k in [80, 160]:                             # 两次清洗后回落
    x[k:] -= 0.45
ax = axes[3]
ax.plot(t, x, color=BLUE, lw=1.4)
ax.plot(t, base, color="gray", lw=1, ls="--", label="无漂移真值")
ax.scatter([80, 160], [x[80], x[160]], s=90, facecolor="none",
           edgecolor=RED, lw=2, zorder=5)
ax.annotate("每次清洗后台阶式回落\n趋势上越漂越高", xy=(175, 1.25),
            ha="center", color=RED, fontsize=9, fontweight="bold")
ax.legend(fontsize=7.5, loc="upper left")
finish(ax, "④量程漂移：越漂越高、清洗回落")

# ⑤ 分析仪冲洗/校准假读数（状态码未联动）
x = base.copy()
for s in [(70, 85), (170, 185)]:
    x[s[0]:s[1]] = np.linspace(0.05, 0.9, s[1]-s[0])
ax = axes[4]
ax.plot(t, x, color=BLUE, lw=1.4)
for s in [(70, 85), (170, 185)]:
    ax.axvspan(s[0], s[1], color=RED, alpha=.13)
ax.annotate("冲洗/换试剂时段\n读数跌到近0，状态位却为正常",
            xy=(128, 0.9), ha="center", color=RED,
            fontsize=9, fontweight="bold")
finish(ax, "⑤冲洗校准假读数：状态码缺失")

# ⑥ 时间戳错乱（一段数据时间戳回跳，折线往回画）
x = base.copy()
ts = t.copy()
seg = slice(90, 130)
ts[seg] = t[seg][::-1]                          # 时间戳倒序
order = np.argsort(ts)
ax = axes[5]
ax.plot(t, x, color=BLUE, lw=1.2, alpha=.55, label="按入库顺序")
ax.plot(ts[seg], x[seg], color=RED, lw=1.6)
ax.scatter([ts[90], ts[129]], [x[90], x[129]], s=90, facecolor="none",
           edgecolor=RED, lw=2, zorder=5)
ax.annotate("时间戳回跳1小时\n折线掉头往回画", xy=(150, 1.55),
            color=RED, fontsize=9, fontweight="bold")
ax.legend(fontsize=7.5, loc="upper left")
finish(ax, "⑥时间戳错乱：时钟漂移/时区")

# ⑦ 单位与4-20mA量程配错
x = base.copy(); x[120:] = x[120:] * 10.0       # 量程系数配错10倍
ax = axes[6]
ax.plot(t[:120], x[:120], color=BLUE, lw=1.4)
ax.plot(t[119:], x[119:], color=RED, lw=1.6)
ax.axhline(5.0, color="orange", ls="--", lw=1.2, label="量程上限5")
ax.scatter([120], [x[120]], s=110, facecolor="none",
           edgecolor=RED, lw=2, zorder=5)
ax.annotate("换表/改量程后整体放大10倍\n读数齐刷刷超限", xy=(196, 9.2),
            ha="center", color=RED, fontsize=9, fontweight="bold")
ax.legend(fontsize=7.5, loc="upper right")
finish(ax, "⑦量程配错：4-20mA换算差10倍")

# ⑧ 多泵运行记录歧义：频率为0，累计流量却在涨
freq = np.where(((t//40) % 2 == 0), 35.0, 0.0)
flow_cum = np.cumsum(np.where(freq > 0, 0.8, 0.02*t*0 + 0.25))  # 停泵期仍在涨
ax = axes[7]
ax.plot(t, flow_cum, color=BLUE, lw=1.5, label="累计投加流量")
ax.fill_between(t, 0, 40, where=freq == 0, color=RED, alpha=.10)
ax2 = ax.twinx()
ax2.step(t, freq, color="gray", lw=1.1, where="post", label="泵运行频率")
ax2.set_ylim(-5, 60); ax2.set_ylabel("频率 Hz", fontsize=8)
ax2.tick_params(labelsize=8)
ax.scatter(t[(freq == 0) & (t > 40)][::12],
           flow_cum[(freq == 0) & (t > 40)][::12],
           s=55, facecolor="none", edgecolor=RED, lw=1.6, zorder=5)
ax.annotate("频率=0停泵期\n累计量照涨：对不上账", xy=(150, 92),
            ha="center", color=RED, fontsize=9, fontweight="bold")
ax.legend(fontsize=7.5, loc="upper left")
ax.set_ylabel("累计流量", fontsize=8)
finish(ax, "⑧多泵歧义：频率与累计量打架")

# ⑨ 人工补录造假：整段同一数字回填
x = base.copy()
x[70:150] = 3.50                               # 批量补录同一个数
ax = axes[8]
ax.plot(t, x, color=BLUE, lw=1.3)
ax.scatter(t[70:150:8], x[70:150:8], s=45, facecolor="none",
           edgecolor=RED, lw=1.5, zorder=5)
ax.annotate("80个补录值全是3.50\n过于整齐，必为批量回填",
            xy=(112, 2.75), ha="center", color=RED,
            fontsize=9, fontweight="bold")
finish(ax, "⑨人工补录：规律数字批量回填")

for ax in axes:
    ax.set_xlabel("时间（分钟）", fontsize=8)
fig.suptitle("脏数据的九种死法：红圈/红底处即为问题（合成数据演示）",
             fontsize=14, fontweight="bold", y=0.995)
plt.tight_layout(rect=[0, 0, 1, 0.97])
out = "../images/fig05_02_dirty_grid.png"
plt.savefig(out)
print("saved:", out)
