"""
图 02-1：干粉药剂配制浓度算例 + 三槽式溶配药装置体积与停留时间
运行：python3 fig02_01_solution_makeup.py
输出：../images/fig02_01_solution_makeup.png
说明：配制 1 m3 溶液的药粉/清水质量关系，PAM 取 0.2%、干粉 PAC 取 10%；
三槽按配药能力 3 m3/h 配置，数字为典型工程算例（具体项目以设备样本为准）。
"""
import matplotlib
matplotlib.use("Agg")
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC"]
plt.rcParams["axes.unicode_minus"] = False

# ---------- 算例 1：配制 1 m3（约 1000 kg）药液 ----------
cases = ["PAM 0.2%\n（典型 0.1%~0.3%）", "干粉 PAC 10%\n（折 Al2O3 计）"]
powder = np.array([2.0, 100.0])    # kg 药粉
water = np.array([998.0, 900.0])   # kg 清水

# ---------- 算例 2：三槽体积与停留时间，配药能力 3 m3/h ----------
tanks = ["溶解槽", "熟化槽", "储药槽"]
volume = np.array([1.5, 3.0, 3.0])      # m3
rate = 3.0                              # m3/h
hrt = volume / rate                     # h

print("=== 配制 1 m3 药液 ===")
for c, p, w in zip(cases, powder, water):
    print(f"{c.replace(chr(10), ' ')}: 药粉 {p:.0f} kg + 清水 {w:.0f} kg")
print("=== 三槽停留时间（配药能力 3 m3/h）===")
for t, v, h in zip(tanks, volume, hrt):
    print(f"{t}: {v:.1f} m3, 停留 {h:.2f} h = {h*60:.0f} min")

fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))

# 左图：药粉与清水堆叠条形
ax = axes[0]
x = np.arange(len(cases))
ax.bar(x, powder, width=0.5, color="#ef6c00", label="药粉质量 kg")
ax.bar(x, water, width=0.5, bottom=powder, color="#1565c0", label="清水质量 kg")
for i, (p, w) in enumerate(zip(powder, water)):
    ax.text(i, p / 2, f"{p:.0f} kg", ha="center", va="center", color="white", fontsize=11)
    ax.text(i, p + w / 2, f"{w:.0f} kg", ha="center", va="center", color="white", fontsize=11)
    ax.text(i, p + w + 25, "溶液合计\n1000 kg ≈ 1 m³", ha="center", va="bottom", fontsize=10)
ax.set_xticks(x)
ax.set_xticklabels(cases)
ax.set_ylabel("质量 kg（每配制 1 m³ 药液）")
ax.set_ylim(0, 1180)
ax.set_title("干粉配药：浓度差 50 倍，药粉用量差 50 倍")
ax.legend(loc="upper right")
ax.grid(axis="y", alpha=.3)

# 右图：三槽体积条形 + 停留时间标注
ax = axes[1]
colors = ["#ffcc80", "#a5d6a7", "#90caf9"]
ax.bar(tanks, volume, width=0.55, color=colors, edgecolor="#555")
for i, (v, h) in enumerate(zip(volume, hrt)):
    ax.text(i, v + 0.08, f"{v:.1f} m³\n停留 {h*60:.0f} min", ha="center", va="bottom", fontsize=11)
ax.axhspan(0.5, 1.0, color="#2e7d32", alpha=.08)
ax.text(2.35, 0.75, "PAM 熟化要求\n30~60 min", ha="right", va="center",
        fontsize=9, color="#2e7d32")
ax.set_ylabel("槽体有效容积 m³")
ax.set_ylim(0, 4.1)
ax.set_title("三槽式溶配药装置（配药能力 3 m³/h 示例）")
ax.grid(axis="y", alpha=.3)

plt.tight_layout()
plt.savefig("../images/fig02_01_solution_makeup.png", dpi=130)
print("saved ../images/fig02_01_solution_makeup.png")
