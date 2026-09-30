"""
图 02-2：计量泵相对流量特性曲线 —— 冲程长度 x 冲程频率 -> 相对流量
运行：python3 fig02_02_pump_characteristic.py
输出：../images/fig02_02_pump_characteristic.png
说明：曲线为合成近似示意（真实曲线以泵厂家出厂测试为准）；
理想情况下相对流量 ≈ 冲程% x 频率%，实际在低端因回流/泄漏略低于理想值。
"""
import matplotlib
matplotlib.use("Agg")
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC"]
plt.rcParams["axes.unicode_minus"] = False

rng = np.random.default_rng(42)

stroke = np.linspace(0, 100, 101)          # 冲程长度 %
freqs = [100, 80, 60, 40, 20]              # 冲程频率 %

def q_ideal(s, f):
    return s * f / 100.0

def q_actual(s, f):
    # 合成近似：低端（小冲程）单向阀关闭不严、容积效率下降，叠加约 1% 离散
    eff = 1.0 - 0.18 * np.clip(1.0 - s / 100.0, 0, 1) ** 2
    return q_ideal(s, f) * eff

print("冲程50%时各频率相对流量（理想/实际）：")
for f in freqs:
    qi, qa = q_ideal(50, f), q_actual(50, f)
    print(f"  频率 {f}%: 理想 {qi:.1f}%  实际约 {qa:.1f}%")

fig, ax = plt.subplots(figsize=(9.5, 6))
colors = ["#2e7d32", "#1565c0", "#ef6c00", "#6a1b9a", "#c62828"]
for f, c in zip(freqs, colors):
    ax.plot(stroke, q_actual(stroke, f), color=c, lw=2,
            label=f"实际 频率 {f}%")
ax.plot(stroke, q_ideal(stroke, 100), color="black", lw=1.4, ls="--",
        label="理想线 频率100%（Q∝冲程）")
ax.scatter([50], [q_actual(50, 50)], color="#c62828", zorder=5)
ax.annotate("例：冲程50% × 频率50%\n实际相对流量约 23%",
            xy=(50, q_actual(50, 50)), xytext=(12, 42),
            arrowprops=dict(arrowstyle="->", color="#c62828"),
            fontsize=10, color="#c62828")
ax.fill_between(stroke, 0, 15, color="#c62828", alpha=.07)
ax.text(7, 12.5, "低端调节区\n计量精度变差", fontsize=9, color="#c62828")
ax.set_xlabel("冲程长度设定 %")
ax.set_ylabel("相对最大流量 %")
ax.set_xlim(0, 100)
ax.set_ylim(0, 102)
ax.set_xticks(range(0, 101, 10))
ax.set_title("计量泵调节特性：相对流量 ≈ 冲程% × 频率%（合成示意曲线）")
ax.legend(loc="upper left", fontsize=9, ncol=2)
ax.grid(alpha=.3)

plt.tight_layout()
plt.savefig("../images/fig02_02_pump_characteristic.png", dpi=130)
print("saved ../images/fig02_02_pump_characteristic.png")
