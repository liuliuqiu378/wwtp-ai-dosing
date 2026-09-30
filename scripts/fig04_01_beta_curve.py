"""
图 04-1：金属盐除磷 投加摩尔比 beta 与出水残余总磷、PAC 单耗的关系（合成经验曲线）
运行：cd scripts && python3 fig04_01_beta_curve.py
输出：../images/fig04_01_beta_curve.png
算例口径：Q=10万 m3/d，进水TP 4.0 mg/L，目标出水TP 0.3 mg/L，PAC 以 Al2O3 29% 计
"""
import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC"]
plt.rcParams["axes.unicode_minus"] = False

rng = np.random.default_rng(42)

# ---- 基本参数（与 04-1 正文算例一致）----
Q = 100000.0          # m3/d
TP_in = 4.0           # mg/L
TP_target = 0.3       # mg/L
dP = TP_in - TP_target                      # 需去除磷 3.7 mg/L
P_kg = Q * dP / 1000.0                      # kg P/d = 370
mol_P = P_kg / 31.0                         # kmol P/d
Al2O3 = 0.29                                # PAC 中 Al2O3 质量分数
w_Al = Al2O3 * (2.0 * 27.0 / 102.0)         # Al 质量分数 ≈ 0.1535
# 单位 beta 对应的 PAC 干粉单耗 (mg/L)
unit_per_beta = (mol_P * 27.0 / w_Al) / Q * 1e3   # mg/L 每 1 单位 beta

beta = np.linspace(0.8, 3.2, 25)
# 合成经验下降曲线：beta=1 约1.60，beta=1.5 约0.66，beta=2 约0.30，beta=3 约0.12
TP_floor = 0.09
TP_res = TP_floor + 1.51 * 0.14 ** (beta - 1.0)
TP_res = TP_res + rng.normal(0, 0.008, beta.size)
pac_dose = unit_per_beta * beta             # mg/L 干粉

fig, ax1 = plt.subplots(figsize=(9.6, 5.6), dpi=130)
l1, = ax1.plot(beta, TP_res, "o-", color="#1f6fb2", lw=2, ms=5, label="出水残余 TP")
ax1.axhline(0.5, color="#888", ls="--", lw=1)
ax1.axhline(0.3, color="#c0392b", ls="--", lw=1.2)
ax1.text(3.05, 0.52, "一级A 0.5", color="#555", fontsize=9, ha="right")
ax1.text(3.05, 0.32, "准IV类 0.3", color="#c0392b", fontsize=9, ha="right")
ax1.scatter([2.0], [TP_target], color="red", zorder=5)
ax1.annotate("算例点 beta=2.0\nTP约0.30 mg/L", xy=(2.0, 0.3), xytext=(2.25, 0.85),
             arrowprops=dict(arrowstyle="->", color="red"), fontsize=9, color="red")
ax1.set_xlabel("实际投加摩尔比 beta = 金属离子摩尔数 / 需沉淀磷摩尔数")
ax1.set_ylabel("出水残余总磷 TP (mg/L)", color="#1f6fb2")
ax1.set_ylim(0, 2.2)
ax1.tick_params(axis="y", labelcolor="#1f6fb2")
ax1.grid(alpha=0.3)

ax2 = ax1.twinx()
l2, = ax2.plot(beta, pac_dose, "s-", color="#e67e22", lw=2, ms=5, label="PAC 干粉单耗")
ax2.set_ylabel("PAC 干粉单耗 (mg/L，Al2O3 29%)", color="#e67e22")
ax2.tick_params(axis="y", labelcolor="#e67e22")
ax2.set_ylim(0, unit_per_beta * 3.2 * 1.15)

ax1.set_title("投加摩尔比 beta 与出水残余总磷、PAC 单耗的经验关系")
lines = [l1, l2]
ax1.legend(lines, [x.get_label() for x in lines], loc="upper center")
plt.tight_layout()
plt.savefig("../images/fig04_01_beta_curve.png")

print("mol_P(kmol/d) =", round(mol_P, 2))
print("w_Al =", round(w_Al, 4))
print("PAC mg/L per beta =", round(unit_per_beta, 2))
for b in (1.0, 1.5, 2.0, 2.5, 3.0):
    tp = 0.09 + 1.51 * 0.14 ** (b - 1.0)
    print(f"beta={b}: TP_res={tp:.2f} mg/L, PAC={unit_per_beta*b:.1f} mg/L, "
          f"干粉={unit_per_beta*b*Q/1000:.0f} kg/d")
