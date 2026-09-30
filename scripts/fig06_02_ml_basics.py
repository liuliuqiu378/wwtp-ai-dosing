"""
图 06-2：机器学习速成两张图（一张图两子图）
上：工业时序必须按时间切分——训练 70% / 验证 15% / 测试 15%
下：模型复杂度 vs 误差——欠拟合、刚刚好、过拟合三段
运行：python3 fig06_02_ml_basics.py
输出：../images/fig06_02_timeseries_split.png（单文件双子图）
说明：全部为合成数据，仅用于演示概念，不代表任何具体水厂。
"""
import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC"]
plt.rcParams["axes.unicode_minus"] = False

rng = np.random.default_rng(42)

# ============ 上子图数据：365 天日 PAC 单耗，含年周期+周周期+噪声 ============
n = 365
day = np.arange(n)
season = 42 + 6*np.sin(2*np.pi*(day - 110)/365)     # 冬夏漂移
week = 2.5*np.sin(2*np.pi*day/7)                     # 周周期
noise = rng.normal(0, 1.2, n)
y = season + week + noise

i1, i2 = int(n*0.70), int(n*0.85)
print("时序切分：训练 0~%d 天，验证 %d~%d 天，测试 %d~%d 天"
      % (i1-1, i1, i2-1, i2, n-1))
print("三段均值分别为 %.2f / %.2f / %.2f mg/L"
      % (y[:i1].mean(), y[i1:i2].mean(), y[i2:].mean()))

# ============ 下子图数据：y = 非线性真函数 + 噪声，比较 1~10 阶多项式 ============
x = rng.uniform(0, 3, 60)
f = lambda v: 1.0 + 2.2*v - 0.9*v**2 + 0.12*v**3
yy = f(x) + rng.normal(0, 0.35, x.size)
ord_ = np.argsort(x)
x, yy = x[ord_], yy[ord_]
i_tr = 40                                                # 前40点训练
x_tr, y_tr = x[:i_tr], yy[:i_tr]
x_te, y_te = x[i_tr:], yy[i_tr:]

degs = np.arange(1, 11)
rmse_tr, rmse_te = [], []
for d in degs:
    poly = PolynomialFeatures(degree=d, include_bias=False)
    Xtr = poly.fit_transform(x_tr[:, None])
    Xte = poly.transform(x_te[:, None])
    m = LinearRegression().fit(Xtr, y_tr)
    rmse_tr.append(np.sqrt(mean_squared_error(y_tr, m.predict(Xtr))))
    rmse_te.append(np.sqrt(mean_squared_error(y_te, m.predict(Xte))))
best = degs[int(np.argmin(rmse_te))]
print("最优多项式阶数 = %d，测试 RMSE = %.3f；10 阶时训练/测试 RMSE = %.3f / %.3f"
      % (best, rmse_te[best-1], rmse_tr[-1], rmse_te[-1]))

# ============ 画图 ============
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7.5), dpi=130)

ax1.plot(day, y, lw=1.1, color="#455a64")
ax1.axvspan(0, i1-1, color="#2e7d32", alpha=.12)
ax1.axvspan(i1, i2-1, color="#f9a825", alpha=.18)
ax1.axvspan(i2, n-1, color="#c62828", alpha=.12)
ax1.text(i1/2, 52, "训练集 70%\n用来学规律", ha="center", fontsize=11,
         color="#1b5e20", fontweight="bold")
ax1.text((i1+i2)/2, 52, "验证集 15%\n用来调参", ha="center", fontsize=11,
         color="#8d6e00", fontweight="bold")
ax1.text((i2+n)/2, 52, "测试集 15%\n模拟未来", ha="center", fontsize=11,
         color="#b71c1c", fontweight="bold")
ax1.axvline(i1, color="gray", ls="--", lw=1)
ax1.axvline(i2, color="gray", ls="--", lw=1)
ax1.set_title("工业时序只能按时间切分，绝不能随机打乱", fontsize=12.5,
              fontweight="bold")
ax1.set_xlabel("一年中的第几天", fontsize=10.5)
ax1.set_ylabel("PAC 单耗（mg/L，合成）", fontsize=10.5)
ax1.set_ylim(30, 55)
ax1.grid(alpha=.3)

ax2.plot(degs, rmse_tr, "o-", color="#1565c0", label="训练误差")
ax2.plot(degs, rmse_te, "s-", color="#c62828", label="测试误差")
ax2.axvline(best, color="#2e7d32", ls="--", lw=1.3)
ax2.text(best+0.15, 0.75, "刚刚好（泛化最好）", color="#2e7d32",
         fontsize=10.5, fontweight="bold")
ax2.annotate("欠拟合：太简单，两头都差", xy=(1.3, 0.6), fontsize=10,
             color="#5d4037")
ax2.annotate("过拟合：训练越来越好，未来越来越差",
             xy=(7.2, 0.62), xytext=(5.2, 0.95), fontsize=10,
             color="#b71c1c", arrowprops=dict(arrowstyle="->", color="#b71c1c"))
ax2.set_xlabel("模型复杂度（多项式阶数）", fontsize=10.5)
ax2.set_ylabel("RMSE（合成数据）", fontsize=10.5)
ax2.set_title("模型复杂度与误差：追求的是测试误差最低，不是训练误差最低",
              fontsize=12.5, fontweight="bold")
ax2.set_xticks(degs)
ax2.legend(fontsize=10.5)
ax2.grid(alpha=.3)

plt.tight_layout()
out = "../images/fig06_02_timeseries_split.png"
plt.savefig(out)
print("saved:", out)
