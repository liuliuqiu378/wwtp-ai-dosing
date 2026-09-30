"""
图 04-4：消毒 CT 双曲线（余氯浓度 vs 接触时间）与商品次氯酸钠单耗直线
运行：cd scripts && python3 fig04_04_ct_curve.py
输出：../images/fig04_04_ct_curve.png
算例：Q=10万 m3/d，有效氯投加 3 mg/L，商品次钠有效氯 10%
"""
import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC"]
plt.rcParams["axes.unicode_minus"] = False

Q = 100000.0
dose = 3.0           # mg/L 以有效氯计
contact = 30.0       # min

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 5.0), dpi=130)

# 左：不同 CT 等级的双曲线
T = np.linspace(5, 90, 200)
for ct0, c in [(10, "#f5b041"), (30, "#c0392b"), (90, "#8e44ad")]:
    ax1.plot(T, ct0 / T, lw=2, color=c, label=f"CT = {ct0} mg·min/L")
# 算例点：3 mg/L x 30 min = 90
ax1.scatter([contact], [dose], color="black", zorder=5)
ax1.annotate("算例 3 mg/L × 30 min\nCT = 90", xy=(30, 3), xytext=(42, 4.2),
             arrowprops=dict(arrowstyle="->"), fontsize=9)
ax1.axvline(30, color="#888", ls="--", lw=1)
ax1.text(30.8, 0.12, "接触时间≥30 min", fontsize=8, color="#555")
ax1.set_xlabel("接触时间 T (min)")
ax1.set_ylabel("接触渠末端残余消毒剂 C (mg/L)")
ax1.set_title("满足目标 CT：浓度与时间可互换（双曲线）")
ax1.legend(fontsize=9)
ax1.grid(alpha=0.3)
ax1.set_ylim(0, 6)

# 右：有效氯投加量 -> 10% 商品次钠用量
d = np.linspace(0.5, 6.0, 50)
eff_kg = d * Q / 1000.0          # 有效氯 kg/d
prod_kg = eff_kg / 0.10          # 10% 商品次钠 kg/d
ax2.plot(d, prod_kg / 1000.0, color="#1f6fb2", lw=2)
ax2.scatter([3.0], [3.0], color="red", zorder=5)
ax2.annotate("算例 3 mg/L\n商品次钠 3.0 t/d\n约 2400 元/d", xy=(3, 3), xytext=(3.8, 1.2),
             arrowprops=dict(arrowstyle="->", color="red"), fontsize=9, color="red")
ax2.set_xlabel("有效氯投加量 (mg/L)")
ax2.set_ylabel("10% 商品次氯酸钠用量 (t/d)")
ax2.set_title("投加量与商品次钠用量为线性关系")
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("../images/fig04_04_ct_curve.png")

print("有效氯需求 =", eff_kg if False else dose * Q / 1000.0, "kg/d")
print("10% 商品次钠 =", dose * Q / 1000.0 / 0.10, "kg/d =",
      dose * Q / 1000.0 / 0.10 / 1000.0, "t/d")
print("按 800 元/t ->", 3.0 * 800, "元/d；折合吨水", 3.0 * 800 / Q, "元/m3")
print("CT 算例 =", dose * contact, "mg·min/L")
