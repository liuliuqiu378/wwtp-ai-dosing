"""
图 05-3：软测量的时间延迟对齐
上：除磷剂投加量 与 出水总磷 两条错峰时序（响应滞后约 3 小时）
下：互相关系数随 lag 的变化，标出最优滞后
运行：python3 fig05_03_softsensor_lag.py
输出：../images/fig05_03_softsensor_lag.png
说明：本图全部为合成数据，仅用于演示互相关找滞后的原理，不代表任何具体水厂。
"""
import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC"]
plt.rcParams["axes.unicode_minus"] = False

rng = np.random.default_rng(42)

dt = 5.0                                  # 5 分钟一个点
hours = 72
warm = 12                                 # 12h 预热，消除卷积启动暂态
total = hours + warm
t_full = np.arange(0, total*60, dt)
th_full = t_full/60.0 - warm              # -12h ~ 72h

TRUE_LAG = 180.0                          # 标注用的水力滞后 180 分钟

# ---- 投加量：非周期阶梯调节（运行员手动调整）+ 3 次错峰进水冲击 ----
dose = np.full_like(th_full, 42.0)
cp = [0, 6, 14, 22, 28, 38, 46, 54, 62, 68]
levels = [42, 48, 40, 52, 44, 49, 41, 53, 45, 47]
for i in range(len(cp)-1):
    dose[(th_full >= cp[i]) & (th_full < cp[i+1])] = levels[i]
dose[th_full >= cp[-1]] = levels[-1]
for pk, amp in [(8, 8), (33, 7), (57, 9)]:
    dose += amp*np.exp(-((th_full-pk)/1.2)**2)
dose += rng.normal(0, 0.7, th_full.size)

# ---- 出水总磷：投加越多出水越低，经 HRT 展宽与滞后 ----
# 单侧指数型停留时间分布核，起点150min、tau30min，均值约180min
kernel_t = np.arange(0, 480, dt)
kernel = np.where(kernel_t >= 150, np.exp(-(kernel_t-150)/30.0), 0.0)
kernel = kernel / kernel.sum()
_full = np.convolve(dose - dose.mean(), kernel, mode="full")
response = _full[:th_full.size]           # 因果对齐：response[m] 只依赖 dose[m-j]
tp_full = 0.35 - 0.0075*response + rng.normal(0, 0.007, th_full.size)

# ---- 去掉预热段 ----
keep = th_full >= 0
th = th_full[keep]
dose = dose[keep]
tp = tp_full[keep]

# ---- 互相关（归一化），lag 范围 -2~10 小时 ----
def cross_corr(x, y, lag_min):
    # lag>0 表示 y（出水总磷）相对 x（投加量）延迟 lag 分钟
    l = int(round(lag_min/dt))
    if l >= 0:
        a, b = x[:x.size-l], y[l:]
    else:
        a, b = x[-l:], y[:y.size+l]
    return np.corrcoef(a, b)[0, 1]

lags = np.arange(-120, 601, 5)
cc = np.array([cross_corr(dose, tp, L) for L in lags])
best = lags[np.argmin(cc)]                # 投加↑→TP↓，取负相关最强点
print("互相关最强滞后 lag =", best, "分钟；相关系数 r =", round(cc.min(), 3))

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), dpi=130,
                               gridspec_kw={"height_ratios": [1.25, 1]})

# ---- 上：双轴错峰时序 ----
ax1.plot(th, dose, color="#c62828", lw=1.5, label="除磷剂投加量")
ax1.set_ylabel("除磷剂投加量（L/h，合成示例）", color="#c62828", fontsize=10)
ax1.tick_params(axis="y", labelcolor="#c62828")
ax1b = ax1.twinx()
ax1b.plot(th, tp, color="#1565c0", lw=1.3, label="出水总磷")
ax1b.set_ylabel("出水总磷（mg/L）", color="#1565c0", fontsize=10)
ax1b.tick_params(axis="y", labelcolor="#1565c0")
pk = 8
ax1.axvline(pk, color="#c62828", ls=":", lw=1)
ax1.axvline(pk + TRUE_LAG/60, color="#1565c0", ls=":", lw=1)
ax1.annotate("", xy=(pk+1.4, 59), xytext=(pk, 59),
             arrowprops=dict(arrowstyle="->", color="k"))
ax1.annotate("", xy=(pk+TRUE_LAG/60, 59), xytext=(pk+1.4, 59),
             arrowprops=dict(arrowstyle="->", color="k"))
ax1.text(pk+1.5, 60.5, "错峰约3h", ha="center", fontsize=10, fontweight="bold")
ax1.set_title("投加量先动，出水总磷约3小时后才响应（HRT决定，合成数据）",
              fontsize=12.5, fontweight="bold")
ax1.grid(alpha=.3)
ax1.set_xlim(0, 72)
l1, lb1 = ax1.get_legend_handles_labels()
l2, lb2 = ax1b.get_legend_handles_labels()
ax1.legend(l1+l2, lb1+lb2, loc="upper right", fontsize=9)

# ---- 下：互相关曲线 ----
ax2.plot(lags/60.0, cc, color="#2e7d32", lw=2)
ax2.axhline(0, color="gray", lw=.8)
ax2.axvline(best/60.0, color="#d32f2f", ls="--", lw=1.4)
ax2.scatter([best/60.0], [cc.min()], s=90, color="#d32f2f", zorder=5)
ax2.annotate(f"最优滞后 lag = {best:.0f} 分钟（{best/60:.1f} 小时）\n"
             f"r = {cc.min():.2f}，负相关最强",
             xy=(best/60.0, cc.min()), xytext=(best/60.0+1.3, cc.min()+0.28),
             fontsize=10.5, color="#d32f2f", fontweight="bold",
             arrowprops=dict(arrowstyle="->", color="#d32f2f"))
ax2.set_xlabel("时间滞后 lag（小时）：出水总磷相对投加量的延迟", fontsize=10.5)
ax2.set_ylabel("互相关系数 r", fontsize=10.5)
ax2.set_title("用互相关自动找出水力停留造成的最优滞后", fontsize=12,
              fontweight="bold")
ax2.grid(alpha=.3)
ax2.set_xlim(-2, 10)
ax1.set_xlabel("时间（小时）", fontsize=10)

plt.tight_layout()
out = "../images/fig05_03_softsensor_lag.png"
plt.savefig(out)
print("saved:", out)
