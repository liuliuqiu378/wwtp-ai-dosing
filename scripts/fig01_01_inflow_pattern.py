"""
图 01-1：一座 10 万吨/天市政厂 24 小时进水流量与 COD 双轴日变化曲线（合成数据）
运行：python3 fig01_01_inflow_pattern.py
输出：../images/fig01_01_inflow_pattern.png
说明：本图为概念性实算演示，参数为典型经验量级，不代表任何具体水厂。
"""
import matplotlib
matplotlib.use("Agg")
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC"]
plt.rcParams["axes.unicode_minus"] = False

rng = np.random.default_rng(42)
t = np.arange(0, 24, 0.25)  # 15 分钟一个点，全天 96 个点

# ---- 进水流量：10 万吨/天厂，平均时流量 4167 m3/h，早/晚双高峰 ----
Q_avg = 100000.0 / 24.0
Q = Q_avg * (1.0
             + 0.26 * np.exp(-((t - 8.5) / 1.9) ** 2)    # 早高峰
             + 0.17 * np.exp(-((t - 19.5) / 2.3) ** 2)   # 晚高峰
             - 0.18 * np.exp(-((t - 3.5) / 2.6) ** 2))   # 夜间低峰
Q = Q + rng.normal(0, 0.015 * Q_avg, t.size)

# ---- 进水 COD：随生活作息同向波动，夜间地下水渗入稀释 ----
cod = (330.0
       + 95.0 * np.exp(-((t - 9.0) / 2.0) ** 2)
       + 55.0 * np.exp(-((t - 20.0) / 2.2) ** 2)
       - 60.0 * np.exp(-((t - 4.0) / 2.8) ** 2)
       + rng.normal(0, 6.0, t.size))

fig, ax1 = plt.subplots(figsize=(10, 5.5), dpi=130)

l1, = ax1.plot(t, Q, color="#1565c0", lw=2.2, label="进水流量")
ax1.axhline(Q_avg, color="#1565c0", ls=":", lw=1.2, alpha=.7)
ax1.text(0.3, Q_avg + 60, "全天平均 4167 m3/h", color="#1565c0", fontsize=9)
ax1.set_xlabel("时间 (h)")
ax1.set_ylabel("进水流量 (m3/h)", color="#1565c0")
ax1.tick_params(axis="y", labelcolor="#1565c0")
ax1.set_xticks(range(0, 25, 2))
ax1.set_xlim(0, 24)
ax1.set_ylim(2500, 5600)
ax1.grid(alpha=.3)

ax2 = ax1.twinx()
l2, = ax2.plot(t, cod, color="#ef6c00", lw=2.2, ls="--", label="进水 COD")
ax2.axhline(cod.mean(), color="#ef6c00", ls=":", lw=1.2, alpha=.7)
ax2.set_ylabel("进水 COD (mg/L)", color="#ef6c00")
ax2.tick_params(axis="y", labelcolor="#ef6c00")
ax2.set_ylim(200, 480)

# 高峰/低峰标注
ax1.annotate("早高峰 8-9 点\n起床洗漱+冲厕", xy=(8.5, Q.max()),
             xytext=(10.2, 5350), fontsize=9,
             arrowprops=dict(arrowstyle="->", color="#555"))
ax1.annotate("晚高峰 19-20 点\n做饭洗澡", xy=(19.5, Q[t >= 18].max()),
             xytext=(20.3, 4900), fontsize=9,
             arrowprops=dict(arrowstyle="->", color="#555"))
ax1.annotate("夜间低峰\n渗入水占比升高", xy=(3.5, Q.min()),
             xytext=(2.5, 2750), fontsize=9,
             arrowprops=dict(arrowstyle="->", color="#555"))

ax1.set_title("一座 10 万吨/天市政厂的进水日变化：流量与 COD 都跟着居民作息走（合成数据）")
ax1.legend(handles=[l1, l2], loc="upper left", fontsize=9)
plt.tight_layout()
plt.savefig("../images/fig01_01_inflow_pattern.png")
print("已保存 ../images/fig01_01_inflow_pattern.png")

# ---- 输出关键结果，供正文引用 ----
Q_day = np.sum(Q) * 0.25
print(f"全天累计进水量: {Q_day:,.0f} m3（设计规模 100000 m3/d）")
print(f"流量: 最小 {Q.min():.0f} / 平均 {Q.mean():.0f} / 最大 {Q.max():.0f} m3/h")
print(f"流量时变化系数 K_h = Qmax/Qavg = {Q.max()/Q.mean():.2f}")
print(f"COD: 最小 {cod.min():.0f} / 平均 {cod.mean():.0f} / 最大 {cod.max():.0f} mg/L")
