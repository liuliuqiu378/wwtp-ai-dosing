"""
图 07-2：除磷加药回路完整仿真——人工经验 / 单 PI / 前馈+PI 三条路线
运行：cd scripts && python3 fig07_02_loop_sim.py
输出：../images/fig07_02_loop_sim.png

对象：10 万吨/天市政厂，PAC 投在生物池出水端（同步沉淀）
      出水 TP 动态为 FOPDT：tau=55 min，theta=45 min（与 04-5 算例同量级）
      稳态化学计量沿用 04-1：beta=2 时 1 mg/L PAC 干粉约除 0.088 mg/L P
仪表：在线 TP 仪 15 min 一个数，噪声 sigma=0.025 mg/L，含跳变尖峰与一段坏值冻结
执行器：计量泵 60~2000 L/h，变化率 <=150 L/h 每 15 min，死区 ±0.02 mg/L
扰动：早晚高峰水量/浓度、夜间低负荷（beta 漂移到约 2.4）、分析仪故障段
"""
import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC"]
plt.rcParams["axes.unicode_minus"] = False

rng = np.random.default_rng(42)

# ---------- 1. 时间与工艺常量 ----------
DT = 15.0                    # min，控制/分析仪周期
DAYS = 3
N = int(DAYS * 24 * 60 / DT)  # 288 拍
WARM = 32                    # 预热 8 h（32 拍），在平稳工况下先跑稳
M = N + WARM
t = np.arange(N) * DT / 60.0             # 小时（记录段从 0 点开始）
hours = (t % 24.0)

Q0 = 100000.0 / 24.0         # m3/h，平均进水流量
C_SOL = 0.10                 # 药液质量分数 10%
RHO = 1.1                    # kg/L，10% PAC 药液密度
K_CHEM = 0.088               # mg/L P 每 mg/L PAC 干粉（beta=2，来自 04-1 算例 3.7/42）
TARGET = 0.30                # 出水 TP 目标 mg/L（准 IV 类口径 0.3）
LIMIT = 0.50                 # 一级 A 红线 mg/L
TAU, THETA = 55.0, 45.0      # min
DELAY = int(round(THETA / DT))
A = np.exp(-DT / TAU)

PUMP_MIN, PUMP_MAX = 60.0, 2000.0       # L/h（04-1 名义工况 1590，泵按约 1.25 倍选型）
RATE_MAX = 150.0                         # L/h 每 15 min
DEADBAND = 0.02                          # mg/L

# ---------- 2. 合成进水场景：水量与投加点前 TP ----------
def gauss(x, mu, sig, amp):
    return amp * np.exp(-0.5 * ((x - mu) / sig) ** 2)

# 水量日变化：早 8 点、晚 19 点两个峰，夜间低谷
Q_rec = Q0 * (1.0 + gauss(hours, 8.5, 2.0, 0.22) + gauss(hours, 19.5, 2.5, 0.18)
              - 0.12 * (hours < 5))
# 投加点前总磷（生物除磷后）：基准 2.5，早峰约 4.0，晚峰约 3.3，夜间约 2.3
day_bias = np.repeat(rng.normal(0, 0.12, DAYS), int(N / DAYS))
TP_rec = (2.5
          + gauss(hours, 8.0, 1.8, 1.5)
          + gauss(hours, 19.0, 2.2, 0.8)
          - 0.2 * (hours < 5)
          + 0.10 * np.sin(2 * np.pi * t / 9.0)       # 缓慢漂移
          + day_bias + rng.normal(0, 0.04, N))
TP_rec = np.clip(TP_rec, 1.8, 4.2)
night_rec = (hours < 5) | (hours > 22.5)
k_rec = np.where(night_rec, K_CHEM * (2.0 / 2.4), K_CHEM)

# 预热段：平稳工况（等价于记录开始前一天下午的平峰）
Q = np.concatenate([np.full(WARM, Q0 * 0.98), Q_rec])
TP_a = np.concatenate([np.full(WARM, 2.6), TP_rec])
k_eff = np.concatenate([np.full(WARM, K_CHEM), k_rec])

# 分析仪故障（按记录段计时）：4 个跳变尖峰；第 2 天早 07:00 起坏值冻结 2 h
spike_idx = set(rng.choice(np.arange(40, N - 40), size=4, replace=False).tolist())
bad_start, bad_len = int((24 + 7.0) * 4), 8

# ---------- 3. 工艺换算与 FOPDT 对象 ----------
def plant_step(y_prev, dose, tp_load, keff):
    """FOPDT 一步：出水 TP 真值。dose=mg/L 干粉，作用与负荷均带 DELAY 拍滞后。"""
    yss = max(0.05, tp_load - keff * dose)
    return A * y_prev + (1 - A) * yss

def mgL_to_Lh(dose_mgL, q_m3h):
    """干粉 mg/L -> 10% 药液 L/h。kg/h = Q*dose/1000；L/h = kg/h/c/rho。"""
    return q_m3h * dose_mgL / 1000.0 / C_SOL / RHO

def Lh_to_mgL(u_lh, q_m3h):
    return u_lh * C_SOL * RHO / q_m3h * 1000.0

# ---------- 4. 分析仪：一阶滤波 + 异常值剔除 + 坏值联锁 ----------
ALPHA = 0.4
class Analyzer:
    """单点突变先剔除；连续两拍读数互相接近则确认是真实阶跃，滤波快速追上；
    连续 3 拍无法确认才判仪表坏值。"""
    def __init__(self):
        self.yf = None
        self.prev = None
        self.hold = 0
        self.bad = False

    def read(self, raw):
        if self.yf is None:
            self.yf = raw
            self.prev = raw
            return raw, False
        if abs(raw - self.yf) > 0.15:
            if self.hold >= 1 and abs(raw - self.prev) <= 0.10:
                self.yf, self.hold, self.bad = raw, 0, False   # 连续确认：真实变化
            else:
                self.hold += 1
                self.bad = self.hold >= 3
            self.prev = raw
            return self.yf, self.bad
        self.hold, self.bad = 0, False
        self.yf = ALPHA * raw + (1 - ALPHA) * self.yf
        self.prev = raw
        return self.yf, False

# ---------- 5. PI 元件：比例 + 分离积分（抗饱和），输出乘性修正系数 ----------
class PICorr:
    """乘性 PI：输出修正系数。积分照常累加，仅在执行器被限幅且误差仍同向时回退本拍积分。"""
    def __init__(self, kp, ti_min, lo, hi):
        self.kp, self.ti, self.lo, self.hi = kp, ti_min, lo, hi
        self.ki_state = 0.0
        self.sat = 0

    def update(self, e, freeze=False):
        if freeze:
            return float(np.clip(1.0 + self.ki_state, self.lo, self.hi))
        dki = self.kp * (DT / self.ti) * e
        if self.sat * e > 0:                 # 上一拍泵被限幅、误差仍同向：本拍不积分
            dki = 0.0
        self.ki_state += dki
        return float(np.clip(1.0 + self.kp * e + self.ki_state, self.lo, self.hi))

# ---------- 6. 三种策略 ----------
def manual_schedule(hour):
    """老师傅一把尺：平时固定 1700 L/h 宁多勿少，6 点半到 10 点早高峰手动加到 1950。"""
    if 6.5 <= hour < 10.0:
        return 1950.0
    return 1700.0

def pump_constrain(u_target, u_prev):
    u = np.clip(u_target, PUMP_MIN, PUMP_MAX)
    u = np.clip(u, u_prev - RATE_MAX, u_prev + RATE_MAX)
    return u

def run(strategy):
    y_rec = np.zeros(N)
    u_rec = np.zeros(N)
    ux = np.zeros(M)
    u_ss = mgL_to_Lh((2.6 - TARGET) / K_CHEM, Q0 * 0.98)   # 预热段稳态投加
    ux[:WARM] = u_ss
    analyzer = Analyzer()
    pi_only = PICorr(kp=0.8, ti_min=240.0, lo=0.5, hi=1.5)
    pi_ff = PICorr(kp=0.6, ti_min=80.0, lo=0.8, hi=1.2)
    y = 0.3
    frozen_at = 0.3
    for j in range(WARM, M):
        k = j - WARM
        jd = j - DELAY
        dose_prev = Lh_to_mgL(ux[jd], Q[jd])
        y = plant_step(y, dose_prev, TP_a[jd], k_eff[jd]) + rng.normal(0, 0.006)

        # 分析仪（仅记录段注入故障）
        raw = y + rng.normal(0, 0.025)
        if k in spike_idx:
            raw += rng.choice([-1.0, 1.0]) * (0.25 + rng.uniform(0, 0.1))
        if bad_start <= k < bad_start + bad_len:
            raw = frozen_at
        yf, bad_spike = analyzer.read(raw)
        bad = bad_spike or (bad_start <= k < bad_start + bad_len)
        if not (bad_start <= k < bad_start + bad_len):
            frozen_at = raw

        hour = hours[k]
        if strategy == "manual":
            u_target = manual_schedule(hour)
        elif strategy == "PI":
            u_nom = mgL_to_Lh(32.5, Q0)      # 固定名义基线偏保守，不随当日流量/浓度补偿
            if bad:
                pi_only.update(0.0, freeze=True)             # 坏值：输出原地保持
                u_target = ux[j - 1]
            else:
                e = 0.0 if abs(yf - TARGET) < DEADBAND else yf - TARGET
                kcorr = pi_only.update(e)
                u_target = u_nom * kcorr
        else:  # FFPI：前馈按当前负荷摩尔比算基础量，PI 仅在 0.8~1.2 内乘性修正
            dose_ff = max(0.0, TP_a[j] - TARGET) / K_CHEM
            u_ff = mgL_to_Lh(dose_ff, Q[j])
            if bad:
                pi_ff.update(0.0, freeze=True)               # 坏值联锁：冻结修正
                u_target = u_ff * float(np.clip(1.0 + pi_ff.ki_state, 0.8, 1.2))
            else:
                e = 0.0 if abs(yf - TARGET) < DEADBAND else yf - TARGET
                kcorr = pi_ff.update(e)
                u_target = u_ff * kcorr
        u_ask = u_target
        ux[j] = pump_constrain(u_target, ux[j - 1])
        # 记录本拍执行器饱和方向，供下一拍抗积分饱和使用
        _sat = 1 if ux[j] < u_ask - 1 else (-1 if ux[j] > u_ask + 1 else 0)
        if strategy == "PI":
            pi_only.sat = _sat
        elif strategy == "FFPI":
            pi_ff.sat = _sat
        y_rec[k] = y
        u_rec[k] = ux[j]
    return y_rec, u_rec

results = {s: run(s) for s in ["manual", "PI", "FFPI"]}

# ---------- 7. 指标统计 ----------
rows = []
for name, label in [("manual", "人工经验"), ("PI", "单 PI"), ("FFPI", "前馈+PI")]:
    y, u = results[name]
    sol_L = u.sum() * DT / 60.0                  # L 药液
    dry_kg = sol_L * RHO * C_SOL                 # kg 干粉
    cost = dry_kg * 2.0                          # 2000 元/t
    over_h = np.sum(y > LIMIT) * DT / 60.0
    iae = np.sum(np.abs(y - TARGET)) * DT / 60.0
    rows.append([label, sol_L / 1000, dry_kg, cost, y.mean(), over_h, iae])

df = pd.DataFrame(rows, columns=["策略", "药液量(m3/3d)", "PAC干粉(kg/3d)",
                                 "药费(元/3d)", "平均出水TP(mg/L)", "超标时长(h)",
                                 "IAE(mg/L·h)"])
base = df.loc[df["策略"] == "人工经验", "PAC干粉(kg/3d)"].iloc[0]
df["相对人工药耗"] = (df["PAC干粉(kg/3d)"] / base * 100).round(1).astype(str) + "%"
df["节药率"] = ((1 - df["PAC干粉(kg/3d)"] / base) * 100).round(1).astype(str) + "%"

# ---------- 8. 出图 ----------
fig, axes = plt.subplots(3, 1, figsize=(10.5, 9.2), sharex=True)
day_ticks = np.arange(0, DAYS * 24 + 1, 6)
colors = {"manual": "#888888", "PI": "#F58518", "FFPI": "#4C78A8"}
labels = {"manual": "人工经验（一把尺）", "PI": "单 PI（反馈）", "FFPI": "前馈+PI"}

ax = axes[0]
ax.plot(t, TP_rec, color="#54A24B", lw=1.4, label="投加点前 TP（负荷）")
ax.set_ylabel("投加点前 TP\nmg/L")
ax.legend(loc="upper right", fontsize=9)
ax.grid(alpha=0.3)
ax.set_title("除磷加药回路三日仿真：负荷扰动与三种策略")

ax = axes[1]
for name in ["manual", "PI", "FFPI"]:
    y, u = results[name]
    ax.plot(t, y, color=colors[name], lw=1.5, label=labels[name])
ax.axhline(LIMIT, color="red", ls="--", lw=1.2)
ax.axhline(TARGET, color="#999999", ls=":", lw=1.2)
ax.text(0.3, LIMIT + 0.03, "一级 A 红线 0.5", color="red", fontsize=9)
ax.text(0.3, TARGET - 0.10, "控制目标 0.3", color="#666666", fontsize=9)
fb0, fb1 = bad_start * DT / 60.0, (bad_start + bad_len) * DT / 60.0
ax.axvspan(fb0, fb1, color="orange", alpha=0.15)
ax.text(fb0 + 0.1, 0.95, "分析仪坏值段", fontsize=8, color="#B36B00")
ax.set_ylabel("出水 TP\nmg/L")
ax.set_ylim(0.0, 1.05)
ax.legend(loc="upper right", fontsize=9, ncol=3)
ax.grid(alpha=0.3)

ax = axes[2]
for name in ["manual", "PI", "FFPI"]:
    _, u = results[name]
    ax.plot(t, u, color=colors[name], lw=1.4, label=labels[name])
ax.axvspan(fb0, fb1, color="orange", alpha=0.15)
ax.set_ylabel("计量泵投加量\nL/h（10%药液）")
ax.set_xlabel("时间（h）")
ax.set_xticks(day_ticks)
ax.legend(loc="upper right", fontsize=9, ncol=3)
ax.grid(alpha=0.3)

plt.tight_layout()
out = "../images/fig07_02_loop_sim.png"
plt.savefig(out, dpi=130)
print("图已输出", out)
print("\n=== 三日仿真指标（合成数据，典型参数实算）===")
print(df.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
