"""
图 08-5：M&V 基线回归与报告期节省（B 厂 PAC，IPMVP 选项A/C 思路）
运行：python3 fig08_05_mv_compare.py
输出：../images/fig08_05_mv_compare.png
说明：基线日数据为合成数据（结构与 03-2 台账一致：年 PAC 约 1251 t、冬高夏低）；
基线模型 PAC = b0 + b1*水量 + b2*进水TP + b3*(20-水温)，用最小二乘拟合；
报告期三个月的真实节药率由脚本按设定控制效果生成，图中红三角落在回归线下方。
"""
import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC"]
plt.rcParams["axes.unicode_minus"] = False

rng = np.random.default_rng(42)

# ---------- 1. 合成基线年 365 天日数据（结构对齐 03-2） ----------
d = np.arange(365)
temp = 18.0 - 10.0 * np.cos(2 * np.pi * (d - 15) / 365)          # 年均约18度，冬8夏28
flow = 10.3 + 1.3 * np.sin(2 * np.pi * (d - 100) / 365) + rng.normal(0, 0.22, 365)  # 万m3/d
tp_in = 4.0 - 0.9 * np.cos(2 * np.pi * (d - 15) / 365) + rng.normal(0, 0.20, 365)
tp_in = np.clip(tp_in, 2.6, 5.8)                                  # 进水总磷 mg/L
# 真实结构：PAC 日耗 t/d
pac = (0.45 + 0.10 * flow + 0.46 * tp_in + 0.06 * (20 - temp)
       + rng.normal(0, 0.12, 365))
pac = np.clip(pac, 1.6, None)
month = (d // 31)[:365]
month = np.clip(((d + 15) // 30.5).astype(int), 0, 11)
df = pd.DataFrame({"month": month, "flow": flow, "tp": tp_in, "temp": temp, "pac": pac})
bm = df.groupby("month").agg(flow=("flow", "sum"), tp=("tp", "mean"),
                             temp=("temp", "mean"), pac=("pac", "sum")).reset_index()

X = np.column_stack([flow, tp_in, 20 - temp])
model = LinearRegression().fit(X, pac)
pred_d = model.predict(X)
r2 = model.score(X, pac)
b0, b1, b2, b3 = (model.intercept_, *model.coef_)
print(f"基线模型 PAC = {b0:.3f} + {b1:.4f}*Q + {b2:.4f}*TP + {b3:.4f}*(20-T) | R2 = {r2:.3f}")
print("基线年 PAC 合计 %.1f t（03-2 台账 1250.9 t）" % pac.sum())
print("基线月均 PAC %.1f t，月均费用 %.2f 万元（单价 1800 元/t）"
      % (pac.sum() / 12, pac.sum() / 12 * 0.18))

bm["pred"] = pd.Series(pred_d).groupby(month).sum().values  # 月基线=当月各日预测之和

# ---------- 2. 报告期三个月（投运后第一年 10/11/12 月） ----------
# （日均水量 万m3/d, 进水TP mg/L, 平均水温, 天数, 控制后真实节药率）
report = [("10月", 10.2, 4.2, 16.5, 31, 0.190),
          ("11月", 9.9, 4.6, 13.0, 30, 0.215),
          ("12月", 10.5, 4.7, 12.0, 31, 0.235)]
rows = []
for mon, q, tp, t, days, rate in report:
    qsum = q * days
    base_pred = model.predict([[q, tp, 20 - t]])[0] * days      # 负荷修正后基线 t/月
    actual = round(base_pred * (1 - rate), 1)
    cut = round(base_pred - actual, 1)
    real_rate = cut / base_pred
    money = cut * 1800 / 10000.0
    rows.append((mon, qsum, tp, t, round(base_pred, 1), actual, cut,
                 round(real_rate * 100, 1), round(money, 2)))
rdf = pd.DataFrame(rows, columns=["月份", "水量万m3", "进水TP", "水温",
                                  "修正后基线t", "报告期实测t", "节药量t",
                                  "节药率%", "折金额万元"])
print(rdf.to_string(index=False))
print("报告期合计：节药 %.1f t，综合节药率 %.1f%%，折金额 %.2f 万元"
      % (rdf["节药量t"].sum(),
         rdf["节药量t"].sum() / rdf["修正后基线t"].sum() * 100,
         rdf["折金额万元"].sum()))

# ---------- 3. 出图 ----------
fig, ax = plt.subplots(figsize=(10.6, 6.2))
ax.scatter(bm["pred"], bm["pac"], s=55, color="#1565c0", zorder=4,
           label="基线期 12 个月实测")
xs = np.linspace(float(bm["pred"].min()) - 8,
                 max(float(bm["pred"].max()), float(rdf["修正后基线t"].max())) + 8, 50)
slope = np.polyfit(bm["pred"], bm["pac"], 1)
ax.plot(xs, np.polyval(slope, xs), color="#1565c0", lw=2,
        label="基线回归线（投运前冻结）", zorder=3)
ax.plot(xs, xs, color="#90a4ae", lw=1.3, ls=":", label="y=x 参考线")

# 报告期点：用修正后基线作 x，实测作 y，落在回归线下方
ax.scatter(rdf["修正后基线t"], rdf["报告期实测t"], marker="^", s=140,
           color="#c62828", zorder=5, label="报告期 10-12 月实测")
band_top = [np.polyval(slope, x) for x in rdf["修正后基线t"]]
for (_, r), top in zip(rdf.iterrows(), band_top):
    ax.vlines(r["修正后基线t"], r["报告期实测t"], top,
              color="#c62828", alpha=0.30, lw=10)
    ax.annotate(f"{r['月份']}", (r["修正后基线t"], r["报告期实测t"]),
                textcoords="offset points", xytext=(-2, -16),
                ha="center", fontsize=9.5, color="#b71c1c")
    ax.annotate(f"省 {r['节药率%']}%", (r["修正后基线t"], top),
                textcoords="offset points", xytext=(-4, 6),
                ha="right", fontsize=9.5, color="#b71c1c")
ax.text(0.98, 0.05, "红色竖向带 = 按负荷修正后的节药量", transform=ax.transAxes,
        fontsize=10, color="#b71c1c", va="bottom", ha="right")

ax.set_xlabel("基线模型预测月 PAC 投加量（t/月，已按当月水量水质水温修正）")
ax.set_ylabel("实际月 PAC 投加量（t/月）")
ax.set_title("B 厂 PAC 基线回归与报告期节省示意（基线为合成数据，具体项目以实测为准）",
             fontsize=12.5)
ax.legend(frameon=False, fontsize=10, loc="upper left")
ax.grid(alpha=.3)
plt.tight_layout()
out = "../images/fig08_05_mv_compare.png"
plt.savefig(out, dpi=130)
print("saved:", out)
