"""
图 02-4：信号链延迟分解（甘特式堆叠条）+ 4-20mA 工程量换算曲线
运行：python3 fig02_04_4_20ma_latency.py
输出：../images/fig02_04_4_20ma_latency.png
说明：延迟数字为典型工程经验值；换算算例取计量泵 0~350 L/h 对应 4~20 mA。
"""
import matplotlib
matplotlib.use("Agg")
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC"]
plt.rcParams["axes.unicode_minus"] = False

# ---------- 延迟环节（秒），典型经验值 ----------
stages = ["探头/采样响应", "分析或变送", "PLC 扫描", "网络传输", "SCADA 归档"]
loop_sec = np.array([3.0, 0.5, 0.2, 0.3, 5.0])        # 秒级回路（DO/pH 类）
loop_ana = np.array([120.0, 1200.0, 0.2, 1.0, 30.0])  # 分析仪回路（总磷类）
print("秒级回路总延迟约 %.1f 秒" % loop_sec.sum())
print("分析仪回路总延迟约 %.0f 秒 ≈ %.1f 分钟" % (loop_ana.sum(), loop_ana.sum() / 60))

# ---------- 4-20mA 换算：泵 0~350 L/h ----------
I = np.linspace(4, 20, 100)
PV = (I - 4) / 16 * 350.0
for ma in [8, 12, 16]:
    print(f"{ma} mA -> {(ma-4)/16*350:.1f} L/h")

fig, axes = plt.subplots(1, 2, figsize=(12, 5.4))

# 左图：堆叠横条（对数轴展示量级差）
ax = axes[0]
colors = ["#90caf9", "#ef9a9a", "#a5d6a7", "#fff59d", "#ce93d8"]
left_s = left_a = 0.0
for st, s, a, c in zip(stages, loop_sec, loop_ana, colors):
    ax.barh(["秒级回路\nDO/pH→泵"], [s], left=left_s, color=c, edgecolor="#555", label=st)
    ax.barh(["分析仪回路\n总磷→泵"], [a], left=left_a, color=c, edgecolor="#555")
    left_s += s
    left_a += a
ax.set_xscale("log")
ax.set_xlabel("从工艺变化到数据落库/指令执行的时间 秒（对数轴）")
ax.set_xlim(0.1, 5000)
ax.set_title("信号链延迟分解：总磷回路比秒级慢约 150 倍")
ax.legend(loc="lower right", fontsize=8)
ax.grid(axis="x", alpha=.3, which="both")
ax.text(loop_sec.sum() * 1.2, -0.42, f"合计约 {loop_sec.sum():.0f} 秒", fontsize=9)
ax.text(loop_ana.sum() / 2, 0.55, f"合计约 {loop_ana.sum()/60:.0f} 分钟",
        fontsize=10, color="#b71c1c", ha="center")

# 右图：4-20mA 换算
ax = axes[1]
ax.plot(I, PV, color="#1565c0", lw=2.2)
ax.axvline(4, color="gray", ls=":", lw=1)
ax.axvline(20, color="gray", ls=":", lw=1)
for ma in [8, 12, 16]:
    val = (ma - 4) / 16 * 350
    ax.scatter([ma], [val], color="#c62828", zorder=5)
    ax.annotate(f"{ma} mA → {val:.0f} L/h", xy=(ma, val), xytext=(ma - 2.4, val + 38),
                fontsize=9, color="#c62828",
                arrowprops=dict(arrowstyle="->", color="#c62828"))
ax.set_xlabel("电流信号 mA")
ax.set_ylabel("工程量：计量泵流量 L/h")
ax.set_xticks(range(4, 21, 2))
ax.set_title("4-20mA 线性换算：PV =（I−4）/16×350 + 0")
ax.grid(alpha=.3)
ax.text(4.1, 330, "4 mA = 0 L/h（下限）", fontsize=9)
ax.text(15.2, 18, "20 mA = 350 L/h（满量程）", fontsize=9)

plt.tight_layout()
plt.savefig("../images/fig02_04_4_20ma_latency.png", dpi=130)
print("saved ../images/fig02_04_4_20ma_latency.png")
