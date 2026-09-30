"""
图 04-2：碳氮比 C/N 对反硝化速率的影响 + 碳源投加量与出水TN、成本的 U 形关系
运行：cd scripts && python3 fig04_02_carbon_denitrification.py
输出：../images/fig04_02_carbon_denitrification.png
算例口径：Q=10万 m3/d，TN 45->12 mg/L，反硝化氮量 3000 kg/d，外加碳源用乙酸钠(COD当量0.78，利用率0.9)
"""
import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC"]
plt.rcParams["axes.unicode_minus"] = False

rng = np.random.default_rng(42)

# ---- 左图：比反硝化速率随 C/N 变化（合成经验曲线，20 摄氏度量级）----
cn = np.linspace(0.5, 8.0, 40)
# 外加碳源时工程量级：平台约 3.2 mg N/(g MLVSS.h)，C/N 约5 后基本饱和
rate = 3.2 * (1 - np.exp(-0.75 * (cn - 0.3))) + 0.15
rate = np.clip(rate, 0.2, None) + rng.normal(0, 0.02, cn.size)

# ---- 右图：乙酸钠投加量 vs 出水TN / 药费 / 综合代价 ----
Q = 100000.0
sodium = np.linspace(0, 20.0, 81)          # t/d 乙酸钠（实物）
util = 0.78 * 0.9                          # 有效 COD 折算
cod_add = sodium * 1000.0 * util           # kg COD/d
# 可反硝化氮量：内源碳承担 6000 kg COD/d，需 5 kg COD/kg N
dn_internal = 6000.0 / 5.0                 # kg N/d = 1200
dn_external = cod_add / 5.0
# 同化带走 3 mg/L（典型2~5）；出水本底约 9.5 mg/L（氨氮3+有机氮等）
TN = 45.0 - 3.0 - (dn_internal + dn_external) * 1000.0 / Q
TN = np.maximum(TN, 9.5)
chem_cost = sodium * 3000.0 / 10000.0      # 万元/d，单价3000元/t
# 超标/提标压力代价：TN 超过控制目标12的部分，按每mg/L 0.8万元/d 计（合成口径）
penalty = np.maximum(TN - 12.0, 0) * 0.8
total = chem_cost + penalty

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.2, 5.0), dpi=130)

ax1.plot(cn, rate, "o-", ms=4, color="#2e7d32")
ax1.axvspan(4.0, 6.0, color="#2e7d32", alpha=0.12)
ax1.text(5.0, 0.015, "工程常用 C/N 4~6", ha="center", fontsize=9, color="#2e7d32")
ax1.set_xlabel("外加碳源 C/N 比 (kg COD / kg NO3-N)")
ax1.set_ylabel("比反硝化速率 (mg N / g MLVSS·h)")
ax1.set_title("反硝化速率随 C/N 比先升后平")
ax1.grid(alpha=0.3)

l1, = ax2.plot(sodium, TN, color="#1f6fb2", lw=2, label="出水 TN")
ax2.axhline(15, color="#888", ls="--", lw=1)
ax2.axhline(12, color="#c0392b", ls="--", lw=1.2)
ax2.text(0.2, 15.4, "一级A 15", fontsize=8, color="#555")
ax2.text(0.2, 12.4, "控制目标 12", fontsize=8, color="#c0392b")
ax2.set_xlabel("乙酸钠投加量 (t/d，实物)")
ax2.set_ylabel("出水 TN (mg/L)", color="#1f6fb2")
ax2.tick_params(axis="y", labelcolor="#1f6fb2")
ax2.set_ylim(8, 47)
ax2.grid(alpha=0.3)

ax3 = ax2.twinx()
l2, = ax3.plot(sodium, chem_cost, color="#e67e22", lw=2, label="碳源药费")
l3, = ax3.plot(sodium, total, color="#8e44ad", lw=2.2, ls="-", label="综合代价 U 形")
ax3.set_ylabel("费用 (万元/d)", color="#8e44ad")
ax3.tick_params(axis="y", labelcolor="#8e44ad")

# 算例点 12.8 t/d -> TN 12
x0 = 12.8
ax2.scatter([x0], [12.0], color="red", zorder=5)
ax2.annotate("算例 12.8 t/d\nTN≈12 mg/L", xy=(x0, 12), xytext=(6.5, 22),
             arrowprops=dict(arrowstyle="->", color="red"), fontsize=9, color="red")
imin = np.argmin(total)
ax3.scatter([sodium[imin]], [total[imin]], color="black", zorder=5)
ax3.annotate("经济最优点", xy=(sodium[imin], total[imin]),
             xytext=(sodium[imin] + 1.2, total[imin] + 0.5), fontsize=9)

ax2.set_title("出水TN、药费与综合代价随投加量变化")
lines = [l1, l2, l3]
ax2.legend(lines, [x.get_label() for x in lines], loc="upper right", fontsize=9)
plt.tight_layout()
plt.savefig("../images/fig04_02_carbon_denitrification.png")

print("反硝化氮量 3000 kg/d；需COD(5.0) =", 3000 * 5.0, "kg/d")
print("内源碳折算 6000 kg COD/d -> 外加 COD =", 15000 - 6000, "kg/d")
print("乙酸钠实物 =", round(9000 / util, 1), "kg/d =", round(9000 / util / 1000, 2), "t/d")
print("药费 =", round(9000 / util * 3.0 / 10000, 2), "万元/d；年费用约",
      round(9000 / util * 3.0 * 365 / 10000, 0), "万元/年")
print("图上经济最优点投加量 =", sodium[imin], "t/d, 综合代价 =", round(total[imin], 2), "万元/d")
print("C/N=5 时比反硝化速率约", round(float(3.2 * (1 - np.exp(-0.75 * 4.7)) + 0.15), 2),
      "mg N/(g MLVSS.h)")
