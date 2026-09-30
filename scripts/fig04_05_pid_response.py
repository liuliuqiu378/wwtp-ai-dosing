"""
图 04-5：加药回路 FOPDT 仿真——手动 / P / PI / 前馈+PI 四种控制对比
运行：cd scripts && python3 fig04_05_pid_response.py
输出：../images/fig04_05_pid_response.png
模型：tau dy/dt = -y + Kp*u(t-theta) + Kd*d(t-theta)
量级：tau=30 min（生物池混合反应），theta=10 min（计量泵+渠道+测量滞后）
场景：t=60 min 进水负荷阶跃 +1（相当于进水TP/TN冲击），看出水与投加量
"""
import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC"]
plt.rcParams["axes.unicode_minus"] = False

rng = np.random.default_rng(42)

# ---------- 1. FOPDT 对象 ----------
tau, theta, Kp, Kd = 30.0, 10.0, -0.8, 1.0   # min, min, 投加通道增益(负), 扰动增益
dt = 1.0                                     # 控制周期 min
n = 480
t = np.arange(n) * dt
delay = int(round(theta / dt))
a = np.exp(-dt / tau)

# 进水可测负荷扰动：60 min 起 +1 阶跃（如早高峰进水浓度冲击）
d = np.where(t >= 60, 1.0, 0.0)


def simulate(controller, noise=0.012):
    """controller(mode): 返回投加 u（偏差量，正=加大投加）"""
    y = np.zeros(n)      # 出水指标偏差，>0 表示超标方向
    u = np.zeros(n)
    e_prev = 0.0
    ui = 0.0             # PI 积分/保持项
    for k in range(1, n):
        ym = y[k - 1] + rng.normal(0, noise)          # 测量值带噪声
        e = ym                                           # 目标=0，e 为出水偏高量
        ku = k - delay
        d_use = d[ku] if ku >= 0 else 0.0
        u_use = u[ku] if ku >= 0 else 0.0

        if controller == "manual":
            ui = 0.0
            u[k] = 0.0
        elif controller == "P":
            ui = 0.0
            u[k] = np.clip(1.5 * e, -2, 3)
        elif controller == "PI":
            ui += 1.2 * dt / 20.0 * e                  # Kc=1.2, Ti=20 min
            ui = np.clip(ui, -2, 3)
            u[k] = np.clip(1.2 * e + ui, -2, 3)
        elif controller == "FFPI":
            # 静态前馈：按可测扰动立即预投；工程上模型总有误差，这里取理想值的90%
            u_ff = 0.9 * (-Kd / Kp) * d[k]
            ui += 0.8 * dt / 25.0 * e                  # 反馈只做修正
            ui = np.clip(ui, -1.5, 1.5)
            u[k] = np.clip(u_ff + 0.8 * e + ui, -2, 3)

        # FOPDT 离散递推（欧拉/指数积分）
        y[k] = a * y[k - 1] + (1 - a) * (Kp * u_use + Kd * d_use)
        e_prev = e
    return y, u


results = {name: simulate(name) for name in ["manual", "P", "PI", "FFPI"]}
labels = {"manual": "手动（固定投加）", "P": "纯比例 P",
          "PI": "比例积分 PI", "FFPI": "前馈 + PI"}
colors = {"manual": "#7f8c8d", "P": "#e67e22", "PI": "#1f6fb2", "FFPI": "#2e7d32"}

# ---------- 2. 评价指标 ----------
print(f"{'控制器':<10}{'最大偏差':>10}{'IAE':>10}{'投加动作量':>12}{'稳态偏差':>10}")
for name, (y, u) in results.items():
    iae = float(np.sum(np.abs(y)) * dt)
    tv = float(np.sum(np.abs(np.diff(u))))           # 投加总动作量（越折腾越大）
    peak = float(np.max(np.abs(y)))
    ss = float(np.mean(y[-60:]))
    print(f"{labels[name]:<12}{peak:>8.3f}{iae:>10.1f}{tv:>12.1f}{ss:>10.3f}")

# ---------- 3. 绘图 ----------
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10.2, 7.4), dpi=130, sharex=True)
for name, (y, u) in results.items():
    ax1.plot(t, y, color=colors[name], lw=2, label=labels[name])
    # 投加量折算成计量泵实际开度感：基准 100 L/h，每偏差1单位=50 L/h
    ax2.plot(t, 100 + 50 * u, color=colors[name], lw=2, label=labels[name])

ax1.axhline(0, color="black", lw=0.8)
ax1.axvspan(60, 70, color="red", alpha=0.08)
ax1.text(65, 0.92, "进水负荷阶跃 +1", color="red", fontsize=9, ha="center")
ax1.set_ylabel("出水指标偏差（>0 为超标方向）")
ax1.set_title("FOPDT 加药回路：四种控制方式抗冲击对比（tau=30 min，theta=10 min）")
ax1.legend(ncol=4, fontsize=9)
ax1.grid(alpha=0.3)

ax2.axvspan(60, 70, color="red", alpha=0.08)
ax2.set_xlabel("时间 (min)")
ax2.set_ylabel("投加量 (L/h，基准100)")
ax2.legend(ncol=4, fontsize=9)
ax2.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("../images/fig04_05_pid_response.png")
print("图已输出 ../images/fig04_05_pid_response.png")
