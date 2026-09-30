"""
图 00-1：人工经验投加 vs AI 前馈投加，24 小时对比（合成数据，10 万吨/天市政厂示例）
运行：python3 fig00_01_dosing_compare.py
输出：../images/fig00_01_dosing_compare.png
说明：本图为概念性实算演示，参数为典型经验量级，不代表任何具体水厂。
"""
import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC"]
plt.rcParams["axes.unicode_minus"] = False

rng = np.random.default_rng(42)
t = np.arange(0, 24, 0.25)  # 15分钟一个点

# ---- 进水侧：日变化规律（早高峰、晚高峰）----
Q_base = 4167.0  # m3/h，10万吨/天的平均时流量
Q = Q_base * (1 + 0.22*np.exp(-((t-8.5)/2.2)**2)
                + 0.15*np.exp(-((t-19.5)/2.4)**2)
                + 0.03*np.sin(t/24*2*np.pi))
TP_in = 4.0 + 2.6*np.exp(-((t-8.0)/1.8)**2) + 1.4*np.exp(-((t-19.0)/2.0)**2) \
        + rng.normal(0, 0.08, t.size)  # 进水总磷 mg/L

# ---- 除磷加药：理论需求随进水总磷负荷变化（见 04-1 机理）----
# 以 10% 剩余质量计，AI 前馈按 Q*TP 负荷 + 出水目标 0.30 mg/L 精细调节
P_load = Q * TP_in / 1000.0  # kg-P/h
dose_need = 35.0 + 9.0 * (P_load - P_load.mean())/P_load.std()  # 任意单位 L/h，仅示意形状
dose_ai = np.clip(dose_need, 18, 78) + rng.normal(0, 0.6, t.size)

# 人工：8:45 看到出水抬升后一把拧大，维持到 16:00 回调；晚高峰再来一次
dose_man = np.full_like(t, 40.0)
dose_man[(t >= 8.75)] = 68.0
dose_man[(t >= 16.0)] = 42.0
dose_man[(t >= 19.75)] = 60.0
dose_man[(t >= 23.0)] = 44.0

# ---- 出水总磷：一阶滞后响应（混合+反应+沉淀，时间常数约1.5h）----
def eff_response(dose, dt=0.25, tau=1.5):
    tp = np.zeros_like(t)
    tp[0] = 0.30
    for i in range(1, t.size):
        target = 0.30 + 0.012*(P_load[i]-P_load.mean()) - 0.0045*(dose[i]-40)
        tp[i] = tp[i-1] + dt/tau*(target - tp[i-1])
    return tp + rng.normal(0, 0.006, t.size)

tp_man = np.clip(eff_response(dose_man), 0, None)
tp_ai = np.clip(eff_response(dose_ai), 0, None)

fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

axes[0].plot(t, Q/1000, color="#1565c0", lw=2)
axes[0].set_ylabel("进水流量 (千m³/h)")
axes[0].set_title("进水侧冲击：早/晚高峰流量与总磷负荷（合成数据）")
axes[0].grid(alpha=.3)
ax0b = axes[0].twinx()
ax0b.plot(t, TP_in, color="#ef6c00", lw=2, ls="--")
ax0b.set_ylabel("进水总磷 (mg/L)", color="#ef6c00")
ax0b.tick_params(axis='y', labelcolor="#ef6c00")

axes[1].plot(t, dose_man, color="#c62828", lw=2, label="人工经验：阶梯式投加")
axes[1].plot(t, dose_ai, color="#2e7d32", lw=2, label="AI前馈：随负荷精细调节")
axes[1].fill_between(t, dose_ai, dose_man, where=dose_man>dose_ai,
                     color="#c62828", alpha=.12, label="多投区（浪费）")
axes[1].set_ylabel("PAC投加量 (L/h，示意)")
axes[1].set_title("加药动作对比：人是『看结果再拧大、忘记调回』，AI 是『提前算、贴着走』")
axes[1].legend(loc="upper right", fontsize=9)
axes[1].grid(alpha=.3)

axes[2].plot(t, tp_man, color="#c62828", lw=2, label="人工操作出水总磷")
axes[2].plot(t, tp_ai, color="#2e7d32", lw=2, label="AI投加出水总磷")
axes[2].axhline(0.5, color="black", ls=":", lw=1.5)
axes[2].axhline(0.3, color="#2e7d32", ls=":", lw=1.2)
axes[2].text(0.2, 0.505, "排放标准 0.5 mg/L（一级A）", fontsize=9)
axes[2].text(0.2, 0.305, "AI 控制目标 0.30 mg/L（留安全裕量）", fontsize=9, color="#2e7d32")
axes[2].set_ylabel("出水总磷 (mg/L)")
axes[2].set_xlabel("时间 (h)")
axes[2].set_xticks(range(0, 25, 2))
axes[2].set_ylim(0, 0.62)
axes[2].legend(loc="upper right", fontsize=9)
axes[2].grid(alpha=.3)

plt.tight_layout()
out = "../images/fig00_01_dosing_compare.png"
plt.savefig(out, dpi=130)
print("saved:", out)

# 量化对比（图上数字就是这样算出来的）
man_total = np.trapezoid(dose_man, t)
ai_total = np.trapezoid(dose_ai, t)
print(f"人工24h累计投加: {man_total:.0f}, AI: {ai_total:.0f}, 节省: {(1-ai_total/man_total)*100:.1f}%")
print(f"人工出水TP峰值: {tp_man.max():.3f}, AI峰值: {tp_ai.max():.3f}")
print(f"人工出水TP均值: {tp_man.mean():.3f}, AI均值: {tp_ai.mean():.3f}")
