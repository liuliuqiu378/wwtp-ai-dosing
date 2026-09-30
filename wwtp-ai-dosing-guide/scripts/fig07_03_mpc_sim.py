"""
图 07-3：纯 numpy 二维输出 MPC 演示——前馈+PI vs 枚举式 MPC
运行：cd scripts && python3 fig07_03_mpc_sim.py
输出：../images/fig07_03_mpc_sim.png

被控变量 2 维：y1=出水 TP(mg/L)，y2=化学污泥瞬时产量(kgDS/h)
操纵变量 1 维：u=PAC 计量泵流量(L/h，10% 药液)
做法（无 scipy/cvxpy）：
  1) 合成伪随机阶跃辨识数据，用 np.linalg.lstsq 拟合线性预测模型
       y1[k+1] = a*y1[k] + b*dose[k-3] + g*TPa[k-3] + c
  2) 每个控制周期在泵量程、变化率约束内，以前馈基准为中心枚举 11 个候选投加量
  3) 假设候选动作在未来 3 拍（45 min）保持不变，滚动预测两个输出；
     未来水量、负荷来自带误差的短期预报，投加浓度按各时刻预报水量折算
  4) 目标 = 软目标偏差罚项 + 超标硬罚项 + 污泥（药耗代理）成本 + 动作变化惩罚，
     并用 DMC 式加性偏差校正（实测减模型预测，滤波后恒定叠加到预测窗）
  5) 只执行第一拍，下一拍重新枚举（滚动优化）
快速反应回路（投加点紧邻正磷酸盐仪）：FOPDT tau=30/theta=15，15 min 周期
      演示工况 beta 固定 2.0（模型匹配），扰动为早晚高峰、测量噪声、3% 负荷预报误差
"""
import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC"]
plt.rcParams["axes.unicode_minus"] = False

rng = np.random.default_rng(42)

# ---------- 1. 公共场景与工艺常量（与 fig07_02 同口径）----------
DT, DAYS = 15.0, 3
N = int(DAYS * 24 * 60 / DT)
WARM = 32
M = N + WARM
t = np.arange(N) * DT / 60.0
hours = t % 24.0
Q0 = 100000.0 / 24.0
C_SOL, RHO, K_CHEM = 0.10, 1.1, 0.088
TARGET, LIMIT = 0.30, 0.50
TAU, THETA = 30.0, 15.0     # 快速回路：投加点紧邻在线正磷酸盐仪，theta 仅仪表周期
DELAY = int(round(THETA / DT))
A = np.exp(-DT / TAU)
PUMP_MIN, PUMP_MAX, RATE_MAX, DEADBAND = 60.0, 2000.0, 150.0, 0.02
SLUDGE_COEF = 0.5                     # kg 化学污泥干固体 / kg PAC 干粉（典型经验 0.4~0.6）

def gauss(x, mu, sig, amp):
    return amp * np.exp(-0.5 * ((x - mu) / sig) ** 2)

Q_rec = Q0 * (1.0 + gauss(hours, 8.5, 2.0, 0.22) + gauss(hours, 19.5, 2.5, 0.18)
              - 0.12 * (hours < 5))
day_bias = np.repeat(rng.normal(0, 0.12, DAYS), int(N / DAYS))
TP_rec = np.clip(2.5 + gauss(hours, 8.0, 1.8, 1.5) + gauss(hours, 19.0, 2.2, 0.8)
                 - 0.2 * (hours < 5) + 0.10 * np.sin(2 * np.pi * t / 9.0)
                 + day_bias + rng.normal(0, 0.04, N), 1.8, 4.2)
# 07-3 演示工况：beta 固定 2.0（模型与对象一致），扰动只保留负荷波动、噪声、预报误差
k_rec = np.full(N, K_CHEM)
Q = np.concatenate([np.full(WARM, Q0 * 0.98), Q_rec])
TP_a = np.concatenate([np.full(WARM, 2.6), TP_rec])
k_eff = np.concatenate([np.full(WARM, K_CHEM), k_rec])
# 负荷/水量预报：当前与历史为实测；未来拍用进水模式预测值（负荷 3%、水量 2% 误差）
TP_fc = TP_a * (1.0 + rng.normal(0, 0.03, TP_a.shape))
Q_fc = Q * (1.0 + rng.normal(0, 0.02, Q.shape))

def mgL_to_Lh(d, q):
    return q * d / 1000.0 / C_SOL / RHO

def Lh_to_mgL(u, q):
    return u * C_SOL * RHO / q * 1000.0

def plant_step(y, dose, tp_load, keff):
    return A * y + (1 - A) * max(0.05, tp_load - keff * dose)

# ---------- 2. 合成辨识：伪随机阶跃投加，最小二乘拟合线性模型 ----------
n_id = 960
load_id = (2.9 + 1.0 * np.sin(2 * np.pi * np.arange(n_id) / 96.0)
           + 0.4 * np.sin(2 * np.pi * np.arange(n_id) / 31.0)
           + rng.normal(0, 0.03, n_id))
# 投加基准跟随负荷（保证线性区），再叠加幅值 6 mg/L 的伪随机阶跃激励
seg = rng.choice([-6.0, -3.0, 3.0, 6.0], size=n_id)
flip = rng.random(n_id) < 0.12
prbs = np.zeros(n_id)
for k in range(1, n_id):
    prbs[k] = seg[k] if flip[k] else prbs[k - 1]
# 上界留出 0.15 mg/L 余量，保证响应不撞零磷下限（非线性会污染线性辨识）
dose_id = np.clip((load_id - TARGET) / K_CHEM + prbs, 8.0,
                  (load_id - 0.15) / K_CHEM)
y_id = np.zeros(n_id) + 0.3
for k in range(1, n_id):
    kd = k - DELAY
    y_id[k] = plant_step(y_id[k - 1], dose_id[kd] if kd >= 0 else dose_id[0],
                         load_id[kd] if kd >= 0 else load_id[0], K_CHEM)
y_id += rng.normal(0, 0.012, n_id)        # 辨识数据带在线仪噪声 sigma=0.012
# y[k+1] 由 y[k]、dose[k+1-DELAY]、TPa[k+1-DELAY] 回归（theta 工程上用互相关估计）
ks = np.arange(DELAY, n_id - 1)
Xd = np.column_stack([y_id[ks], dose_id[ks + 1 - DELAY],
                      load_id[ks + 1 - DELAY], np.ones_like(ks, dtype=float)])
yd = y_id[ks + 1]
coef, *_ = np.linalg.lstsq(Xd, yd, rcond=None)
a_id, b_id, g_id, c_id = coef
fit_rmse = float(np.sqrt(np.mean((yd - Xd @ coef) ** 2)))
print("辨识模型 y[k+1] = %.4f*y[k] %+.5f*dose[k-%d] %+.4f*TPa[k-%d] %+.4f"
      % (a_id, b_id, DELAY, g_id, DELAY, c_id))
print("理论参照 a=%.4f b=%.5f g=%.4f；拟合 RMSE=%.4f mg/L"
      % (A, (1 - A) * (-K_CHEM), 1 - A, fit_rmse))

# ---------- 3. 分析仪与前馈PI（参数与 fig07_02 一致）----------
ALPHA = 0.4
class Analyzer:
    """单点突变先剔除；连续两拍互相接近则确认为真实阶跃并快速追上。"""
    def __init__(self):
        self.yf = None
        self.prev = None
        self.hold = 0
    def read(self, raw):
        if self.yf is None:
            self.yf, self.prev = raw, raw
            return raw
        if abs(raw - self.yf) > 0.15:
            if self.hold >= 1 and abs(raw - self.prev) <= 0.10:
                self.yf, self.hold = raw, 0
            else:
                self.hold += 1
            self.prev = raw
            return self.yf
        self.hold = 0
        self.yf = ALPHA * raw + (1 - ALPHA) * self.yf
        self.prev = raw
        return self.yf

class PICorr:
    def __init__(self, kp, ti, lo, hi):
        self.kp, self.ti, self.lo, self.hi = kp, ti, lo, hi
        self.ki = 0.0
        self.sat = 0
    def update(self, e):
        dki = self.kp * (DT / self.ti) * e
        if self.sat * e > 0:
            dki = 0.0
        self.ki += dki
        return float(np.clip(1.0 + self.kp * e + self.ki, self.lo, self.hi))

def constrain(u_target, u_prev):
    return float(np.clip(np.clip(u_target, PUMP_MIN, PUMP_MAX),
                         u_prev - RATE_MAX, u_prev + RATE_MAX))

# MPC 权重：软偏差 / 超标硬罚 / 污泥成本（药耗代理）/ 动作变化
H, N_CAND = 3, 11
W_SOFT, W_HARD, W_LIN, W_SLUDGE, W_DU = 25.0, 3000.0, 200.0, 0.030, 5.0e-5

def mpc_cost(u_cand, y_now, j_now, ux, bias):
    """候选动作保持 H 拍：h<DELAY 受管存历史动作影响，h>=DELAY 候选动作生效。"""
    yp = y_now
    j_cost = 0.0
    for h in range(1, H + 1):
        jf = j_now + h - DELAY
        if jf <= j_now - 1:
            dose_h = Lh_to_mgL(ux[jf], Q[jf])      # 还在管道里的历史动作
            u_eff = ux[jf]
        else:
            u_eff = u_cand
            dose_h = Lh_to_mgL(u_eff, Q_fc[min(jf, M - 1)])  # 固定药液流量，浓度随未来水量变
        load_use = TP_fc[min(jf, M - 1)] if jf > j_now else TP_a[min(jf, M - 1)]
        yp = max(0.03, a_id * yp + b_id * dose_h + g_id * load_use + c_id) + bias
        sludge = SLUDGE_COEF * u_eff * C_SOL * RHO
        j_cost += (W_SOFT * (yp - TARGET) ** 2
                   + W_HARD * max(0.0, yp - LIMIT) ** 2
                   + W_LIN * max(0.0, yp - LIMIT)
                   + W_SLUDGE * sludge) * (DT / 60.0)
    j_cost += W_DU * (u_cand - ux[j_now - 1]) ** 2
    return j_cost

# ---------- 4. 两个控制器跑同一对象 ----------
def run(mode):
    y_rec, u_rec = np.zeros(N), np.zeros(N)
    ux = np.zeros(M)
    ux[:WARM] = mgL_to_Lh((2.6 - TARGET) / K_CHEM, Q0 * 0.98)
    az = Analyzer()
    pi = PICorr(0.6, 80.0, 0.8, 1.2)
    y = 0.3
    yhat, bias = 0.3, 0.0          # 模型预测状态与 DMC 加性偏差校正
    g_bias = 0.20                  # 偏差滤波增益（现场 beta 漂移时取 0.1~0.3）
    for j in range(WARM, M):
        k = j - WARM
        jd = j - DELAY
        y = plant_step(y, Lh_to_mgL(ux[jd], Q[jd]), TP_a[jd], k_eff[jd]) \
            + rng.normal(0, 0.006)
        yf = az.read(y + rng.normal(0, 0.025))
        # 预测偏差校正：实测值与模型一步预测之差，滤波后作为未来预测的恒定偏置
        # DMC 加性校正：模型对当前拍的开环预测与实测之差，滤波后恒定叠加到预测窗
        yp_open = (a_id * yhat + b_id * Lh_to_mgL(ux[jd], Q[jd])
                   + g_id * TP_a[jd] + c_id)
        bias = g_bias * (yf - yp_open) + (1.0 - g_bias) * bias
        yhat = yp_open + bias
        e = 0.0 if abs(yf - TARGET) < DEADBAND else yf - TARGET
        u_ff = mgL_to_Lh(max(0.0, TP_a[j] - TARGET) / K_CHEM, Q[j])

        if mode == "FFPI":
            u_ask = u_ff * pi.update(e)
        else:  # MPC：在约束网格内枚举 11 个候选，取 H 步目标最小者
            # 候选以前馈基准为中心细化（步长 5 L/h），受泵量程与变化率约束
            lo = max(PUMP_MIN, u_ff - 25.0, ux[j - 1] - RATE_MAX)
            hi = min(PUMP_MAX, u_ff + 25.0, ux[j - 1] + RATE_MAX)
            cands = np.linspace(lo, hi, N_CAND)
            u_ask = cands[int(np.argmin([mpc_cost(c, yf, j, ux, bias) for c in cands]))]
        ux[j] = constrain(u_ask, ux[j - 1])
        pi.sat = 1 if ux[j] < u_ask - 1 else (-1 if ux[j] > u_ask + 1 else 0)
        y_rec[k], u_rec[k] = y, ux[j]
    return y_rec, u_rec

res = {m: run(m) for m in ["FFPI", "MPC"]}

# ---------- 5. 指标 ----------
rows = []
for mode, lab in [("FFPI", "前馈+PI"), ("MPC", "枚举式MPC")]:
    yy, uu = res[mode]
    dry = uu.sum() * DT / 60.0 * RHO * C_SOL      # kg 干粉
    rows.append([lab, dry / 1000, dry * 2.0 / 1000, dry * SLUDGE_COEF / 1000,
                 yy.mean(), np.sum(yy > LIMIT) * DT / 60.0,
                 np.sum(np.abs(yy - TARGET)) * DT / 60.0,
                 uu.sum() * DT / 60.0 / 1000.0])
df = pd.DataFrame(rows, columns=["策略", "PAC干粉(t/3d)", "药费(万元/3d)",
                                 "化学污泥(tDS/3d)", "平均出水TP", "超标时长(h)",
                                 "IAE(mg/L·h)", "药液量(m3/3d)"])
base = df.loc[df["策略"] == "前馈+PI", "PAC干粉(t/3d)"].iloc[0]
df["相对药耗"] = (df["PAC干粉(t/3d)"] / base * 100).round(1).astype(str) + "%"

# ---------- 6. 出图 ----------
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10.5, 7.4), sharex=True)
for mode, lab, c in [("FFPI", "前馈+PI", "#4C78A8"), ("MPC", "枚举式 MPC", "#E45756")]:
    yy, uu = res[mode]
    ax1.plot(t, yy, lw=1.5, label=lab, color=c)
    ax2.plot(t, uu, lw=1.4, label=lab, color=c)
ax1.axhline(LIMIT, color="red", ls="--", lw=1.1)
ax1.axhline(TARGET, color="#888888", ls=":", lw=1.1)
ax1.text(0.5, LIMIT + 0.03, "一级 A 红线 0.5", color="red", fontsize=9)
ax1.set_ylabel("出水 TP mg/L")
ax1.set_ylim(0, 1.0)
ax1.legend(loc="upper right", fontsize=10)
ax1.grid(alpha=0.3)
ax1.set_title("二维输出 MPC（出水TP + 化学污泥）与前馈+PI 三日仿真对比")
ax2.set_ylabel("PAC 计量泵 L/h")
ax2.set_xlabel("时间 h")
ax2.set_xticks(np.arange(0, DAYS * 24 + 1, 6))
ax2.legend(loc="upper right", fontsize=10)
ax2.grid(alpha=0.3)
plt.tight_layout()
out = "../images/fig07_03_mpc_sim.png"
plt.savefig(out, dpi=130)
print("图已输出", out)
print("\n=== 三日仿真指标（合成数据，典型参数实算）===")
print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
