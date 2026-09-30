"""
图 06-7：影子运行 28 天人机建议对比（15分钟合成数据）
AI只记录建议、不执行；对比运行员实际投加，统计人机偏差与假设节药
运行：python3 fig06_07_shadow.py
输出：../images/fig06_07_shadow_compare.png
说明：合成数据，节药幅度按04篇口径（10%药液密度1.1、PAC 2000元/t）折算。
"""
import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC"]
plt.rcParams["axes.unicode_minus"] = False

rng = np.random.default_rng(42)

# ============ 28天、15分钟 ============
N = 28*96
dt = pd.date_range("2025-11-05", periods=N, freq="15min")
doy = dt.dayofyear.to_numpy()
hour = dt.hour + dt.minute/60.0
temp = 18.0 - 10.0*np.cos(2*np.pi*(doy - 15)/365.0)
diurnal = 0.10*np.sin(2*np.pi*(hour-9)/24) + 0.08*np.sin(4*np.pi*(hour-9)/24)
ar = np.zeros(N)
for i in range(1, N):
    ar[i] = 0.985*ar[i-1] + rng.normal(0, 0.012)
Q = 4167.0*(1.0 + diurnal + ar)
tp_in = np.array(np.clip(4.0 + 0.55*np.sin(2*np.pi*(hour-8)/24)
                         + rng.normal(0, 0.12, N), 1.4, 8.0))
Q = np.asarray(Q, dtype=float)
cold = 1.0/(1.0 + np.exp((temp - 13.0)/2.2))
beta = 2.0 + 0.22*cold + 0.35*np.clip((4.0-tp_in)/2.5, 0, 1.0) \
       + rng.normal(0, 0.015, N)
dose_req = Q*(tp_in-0.30)/1000.0*(27/31)*beta/0.1535/0.10/1.1
dose_req = dose_req*(1.0 + rng.normal(0, 0.02, N))

# AI建议：机理需要量+5%安全余量+2.5%模型误差
ai = dose_req*(1.05 + rng.normal(0, 0.025, N))
ai = np.clip(ai, 300, 2800)

# 运行员：8h一块，块首看表设定=当下需要量x(1.15~1.25)，量化到50 L/h
block = 32
op = np.zeros(N)
panic_blocks = rng.choice(np.arange(2, N//block-1), size=8, replace=False)
panic_set = set(panic_blocks.tolist())
for b in range(0, N//block):
    i = b*block
    factor = 1.15 + rng.uniform(0, 0.10)
    if b in panic_set:                    # 出水TP报警后猛加一块药
        factor = rng.uniform(1.35, 1.55)
    val = np.round(dose_req[i]*factor/50)*50
    op[b*block:(b+1)*block] = val

diff = op - ai
print("=== 28天影子运行（AI只建议不执行）===")
print("运行员均值 %.0f L/h，AI建议均值 %.0f L/h，运行员平均偏高 %.1f%%"
      % (op.mean(), ai.mean(), 100*(op.mean()/ai.mean() - 1)))
print("人机偏差超过10%%的时段占 %.1f%%；AI高于人工的时段占 %.1f%%"
      % (100*(np.abs(diff)/ai > 0.10).mean(),
         100*(ai > op).mean()))

# 节药折算（0.25h/点；10%药液 -> 干粉x0.10x1.1 kg/L；2000元/t）
saved_L = (diff*0.25).cumsum()
saved_t = saved_L*0.10*1.1/1000.0
op_t = (op*0.25).sum()*0.10*1.1/1000.0
ai_t = (ai*0.25).sum()*0.10*1.1/1000.0
saved_cost = saved_t*2000/10000.0          # 万元
print("\n28天运行员干粉 %0.1f t，AI建议 %0.1f t，假设节省 %0.1f t，约 %.2f 万元"
      % (op_t, ai_t, op_t-ai_t, (op_t-ai_t)*2000/10000))
print("节省比例 %.1f%%；年化节省约 %.0f t、%.0f 万元"
      % (100*(1-ai_t/op_t),
         (op_t-ai_t)*365/28, (op_t-ai_t)*2000/10000*365/28))

# ============ 画图：上=前3天人机曲线+偏差填充；下=累计假设节药 ============
fig, (a1, a2) = plt.subplots(2, 1, figsize=(11.5, 7), dpi=130,
                             gridspec_kw={"height_ratios": [1.15, 1.0]})
z = 3*96
xx = np.arange(z)/96.0
a1.plot(xx, op[:z], color="#8e24aa", lw=1.6, label="运行员实际投加（8h阶梯）")
a1.plot(xx, ai[:z], color="#2e7d32", lw=1.4, label="AI建议（影子，不执行）")
a1.fill_between(xx, op[:z], ai[:z], where=op[:z] >= ai[:z],
                color="#ffcdd2", alpha=.6, label="多投的药")
a1.set_title("影子运行前3天：运行员长期高一块，AI曲线贴着真实需要走",
             fontsize=12.5, fontweight="bold")
a1.set_ylabel("PAC药液流量（L/h）")
a1.set_xlabel("天数")
a1.legend(fontsize=9.5, loc="upper right")
a1.grid(alpha=.3)

days = np.arange(N)/96.0
a2.plot(days, saved_cost, color="#c62828", lw=2)
a2.fill_between(days, 0, saved_cost, color="#ffcdd2", alpha=.5)
a2.scatter([27], [saved_cost[-1]], s=60, color="#c62828", zorder=5)
a2.annotate("28天假设节药 %.2f 万元\n年化约 %.0f 万元"
            % (saved_cost[-1], saved_cost[-1]*365/28),
            xy=(27, saved_cost[-1]), xytext=(18, saved_cost[-1]*0.55),
            fontsize=11, fontweight="bold", color="#b71c1c",
            arrowprops=dict(arrowstyle="->", color="#b71c1c"))
a2.set_title("累计假设节药金额（只记账，不动泵）", fontsize=12.5,
             fontweight="bold")
a2.set_xlabel("影子运行天数")
a2.set_ylabel("累计假设节省药费（万元）")
a2.grid(alpha=.3)
plt.tight_layout()
plt.savefig("../images/fig06_07_shadow_compare.png")
plt.close()
print("\nsaved: ../images/fig06_07_shadow_compare.png")
