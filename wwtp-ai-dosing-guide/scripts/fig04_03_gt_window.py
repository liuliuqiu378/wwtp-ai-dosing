"""
图 04-3：混凝三阶段（快速混合/絮凝/沉淀）的 G 值区间与 GT 工作窗口
运行：cd scripts && python3 fig04_03_gt_window.py
输出：../images/fig04_03_gt_window.png
算例：水温20摄氏度 mu=1.0e-3 Pa.s；快混 V=20 m3 G=700；絮凝 V=100 m3 G=40 T=20min
"""
import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC"]
plt.rcParams["axes.unicode_minus"] = False

# ---- 算例（与 04-3 正文一致）----
mu = 1.0e-3          # Pa.s，20 摄氏度清水黏度
V_fast, G_fast, t_fast = 20.0, 700.0, 60.0       # m3, s-1, s
V_floc, G_floc, t_floc = 100.0, 40.0, 1200.0     # m3, s-1, s
P_fast = G_fast**2 * mu * V_fast                 # W
P_floc = G_floc**2 * mu * V_floc                 # W
GT_fast, GT_floc = G_fast * t_fast, G_floc * t_floc

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 5.2), dpi=130)

# 左：三阶段 G 值区间条
stages = ["快速混合", "絮凝反应", "沉淀区"]
g_lo = [500, 20, 0.2]
g_hi = [1000, 70, 5.0]
colors = ["#c0392b", "#e67e22", "#2e7d32"]
ypos = [3, 2, 1]
for y, lo, hi, c, name in zip(ypos, g_lo, g_hi, colors, stages):
    ax1.barh(y, hi - lo, left=lo, height=0.5, color=c, alpha=0.55)
    ax1.text(hi * 1.15, y, f"{lo:g}~{hi:g}", va="center", fontsize=9)
ax1.scatter([G_fast, G_floc], [3, 2], color="black", zorder=5)
ax1.annotate("算例 700", xy=(G_fast, 3), xytext=(420, 3.45), fontsize=9)
ax1.annotate("算例 40", xy=(G_floc, 2), xytext=(95, 2.45), fontsize=9)
ax1.set_xscale("log")
ax1.set_yticks(ypos)
ax1.set_yticklabels(stages)
ax1.set_xlabel("速度梯度 G (s-1，对数轴)")
ax1.set_title("混凝三阶段 G 值经验区间")
ax1.grid(alpha=0.3, axis="x")

# 右：GT 工作窗口（G-T 对数图 + GT 等值线）
G = np.logspace(0, 3.2, 200)
for gt0, ls0 in [(1e4, "--"), (1e5, "--")]:
    ax2.plot(G, gt0 / G, color="#888", ls=ls0, lw=1)
    ax2.text(1300, gt0 / 1300 * 1.05, f"GT={gt0:.0e}".replace("e+0", "e"), fontsize=8, color="#555")

# 快混窗口：G 500~1000，t 30~120 s
ax2.fill_between([500, 1000], 30, 120, color="#c0392b", alpha=0.18)
ax2.text(690, 150, "快速混合区", color="#c0392b", fontsize=9, ha="center")
# 絮凝窗口：G 20~70，t 600~1800 s
ax2.fill_between([20, 70], 600, 1800, color="#e67e22", alpha=0.22)
ax2.text(38, 1950, "絮凝反应区", color="#e67e22", fontsize=9, ha="center")
ax2.scatter([G_fast, G_floc], [t_fast, t_floc], color="black", zorder=5)
ax2.annotate("算例 700x60s\nGT=4.2e4", xy=(G_fast, t_fast), xytext=(260, 90),
             arrowprops=dict(arrowstyle="->"), fontsize=9)
ax2.annotate("算例 40x1200s\nGT=4.8e4", xy=(G_floc, t_floc), xytext=(90, 900),
             arrowprops=dict(arrowstyle="->"), fontsize=9)
ax2.set_xscale("log"); ax2.set_yscale("log")
ax2.set_xlabel("G (s-1)")
ax2.set_ylabel("水力停留时间 T (s)")
ax2.set_title("GT 经验窗口 1e4 ~ 1e5")
ax2.grid(alpha=0.3, which="both")

plt.tight_layout()
plt.savefig("../images/fig04_03_gt_window.png")

print(f"快混 P = {P_fast:.0f} W = {P_fast/1000:.1f} kW, GT = {GT_fast:.0f}")
print(f"絮凝 P = {P_floc:.0f} W, GT = {GT_floc:.0f}")
print("若水温降至10摄氏度 mu=1.31e-3，同样功率G_fast =",
      round(np.sqrt(P_fast / (1.31e-3 * V_fast)), 0), "s-1")
