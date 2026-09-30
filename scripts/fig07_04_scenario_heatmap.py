"""
图 07-4：轻量"策略回放台"——六种进水/设备场景下回放三种加药策略
运行：cd scripts && python3 fig07_04_scenario_heatmap.py
输出：../images/fig07_04_scenario_heatmap.png

定位：BSM 类平台的"轻量平替"。用同一套合成场景日（96 拍，15 min），
      把人工经验、前馈+PI、枚举式 MPC 三条策略放进同一条 FOPDT 除磷回路回放，
      横向比较药耗、超标时长与流量加权出水磷负荷（BSM EQI 口径的轻量代理），
      再把药耗与水质风险折算成 0 至 100 的综合得分（分越低越好）。

对象与 fig07_02 同口径：tau=55 min、theta=45 min（慢回路），beta=2 设计工况
      低温季对象化学效率下降 20%（beta 实际约 2.5）：前馈+PI 靠积分慢慢适应，
      MPC 使用换季再辨识后的模型（模拟现场每周滚动辨识）
场景（每类 1 天，96 拍；预热 32 拍平稳段消除初值）：
  1 旱平日   早晚双峰的标准市政负荷
  2 降雨日   10 至 16 点流量 +45%、浓度先冲顶后稀释（初期雨水）
  3 节假日   水量 -22%，峰幅减半且整体后移 2 h
  4 低温季   化学除磷效率 -20%、背景负荷 +0.3（模型失配，MPC 换季再辨识）
  5 工业冲击 13 至 16 点排口 TP 冲到 5.5，MPC 有企业排产预报可提前布防
  6 仪表漂移 13 至 17 点在线 TP 仪缓漂 +0.30，MPC 用模型残差限幅抵御并报警
"""
import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC"]
plt.rcParams["axes.unicode_minus"] = False

rng = np.random.default_rng(42)

# ---------- 1. 工艺常量（与 fig07_02/03 同口径）----------
DT, WARM = 15.0, 32
ND = 96                                   # 一天 96 拍
M = ND + WARM
hours = np.arange(ND) * DT / 60.0
Q0 = 100000.0 / 24.0
C_SOL, RHO, K_CHEM = 0.10, 1.1, 0.088
TARGET, LIMIT = 0.30, 0.50
TAU, THETA = 55.0, 45.0
DELAY = int(round(THETA / DT))
A = np.exp(-DT / TAU)
PUMP_MIN, PUMP_MAX, RATE_MAX, DEADBAND = 60.0, 2000.0, 150.0, 0.02
SLUDGE_COEF = 0.5

def gauss(x, mu, sig, amp):
    return amp * np.exp(-0.5 * ((x - mu) / sig) ** 2)

def mgL_to_Lh(d, q):
    return q * d / 1000.0 / C_SOL / RHO

def Lh_to_mgL(u, q):
    return u * C_SOL * RHO / q * 1000.0

def plant_step(y, dose, tp_load, keff):
    return A * y + (1.0 - A) * max(0.05, tp_load - keff * dose)

# ---------- 2. 六个场景日的流量、投加点前 TP、化学效率、预报、仪表漂移 ----------
# 六个场景共用同一份基准日（固定随机种子序列），场景之间只差各自的扰动，便于横向对比
Q_BASE = Q0 * (1.0 + gauss(hours, 8.5, 2.0, 0.22) + gauss(hours, 19.5, 2.5, 0.18)
               - 0.12 * (hours < 5))
TP_BASE = np.clip(2.5 + gauss(hours, 8.0, 1.8, 1.5) + gauss(hours, 19.0, 2.2, 0.8)
                  - 0.2 * (hours < 5) + 0.10 * np.sin(2 * np.pi * hours / 9.0)
                  + rng.normal(0, 0.04, ND), 1.8, 4.2)
# 典型日模式（确定性日内形状），用于 MPC 的短期负荷预报
TP_pat = np.clip(2.5 + gauss(hours, 8.0, 1.8, 1.5) + gauss(hours, 19.0, 2.2, 0.8)
                 - 0.2 * (hours < 5) + 0.10 * np.sin(2 * np.pi * hours / 9.0), 1.8, 4.2)

def make_scenario(name):
    q, tp = Q_BASE.copy(), TP_BASE.copy()
    keff = np.full(ND, K_CHEM)
    fc_known = False
    drift = np.zeros(ND)
    sr = np.random.default_rng({"旱平日": 1, "降雨日": 2, "节假日": 3,
                                "低温季": 4, "工业冲击": 5, "仪表漂移": 6}[name])
    if name == "旱平日":
        pass
    elif name == "降雨日":
        rain = ((hours >= 10) & (hours < 16)).astype(float)
        q = q * (1.0 + 0.45 * rain)
        tp = tp - 0.9 * rain                       # 稀释
        tp[(hours >= 10) & (hours < 11)] += 1.0   # 初期雨水冲顶
        tp = np.clip(tp, 1.0, 5.0)
    elif name == "节假日":
        q *= 0.78
        q = 0.5 * q + 0.5 * np.interp((hours - 2.0) % 24.0, hours, q)
        tp = 2.2 + (tp - 2.5) * 0.5
        tp = np.interp((hours - 2.0) % 24.0, hours, tp)
        tp += sr.normal(0, 0.03, ND)
        tp = np.clip(tp, 1.6, 3.6)
    elif name == "低温季":
        keff = np.full(ND, K_CHEM * 0.80)         # 低温：化学效率 -20%
        tp = np.clip(tp + 0.30, 1.8, 4.5)
    elif name == "工业冲击":
        ind = ((hours >= 13) & (hours < 16)).astype(float)
        tp = np.clip(tp + 2.3 * ind, 1.8, 5.8)
        q = q * (1.0 + 0.15 * ind)
        fc_known = True                            # 企业排产已报备，MPC 可知
    elif name == "仪表漂移":
        ramp = np.clip((hours - 13.0) / 2.0, 0, 1) - np.clip((hours - 17.0) / 0.5, 0, 1)
        drift = 0.30 * np.clip(ramp, 0, 1)
    else:
        raise ValueError(name)
    return q, np.clip(tp, 0.5, 7.0), keff, fc_known, drift

SCENARIOS = ["旱平日", "降雨日", "节假日", "低温季", "工业冲击", "仪表漂移"]

# ---------- 3. 分析仪（fig07_02 同款）与乘性 PI ----------
ALPHA = 0.4
class Analyzer:
    def __init__(self):
        self.yf = self.prev = None
        self.hold = 0
    def read(self, raw):
        if self.yf is None:
            self.yf = self.prev = raw
            return raw
        if abs(raw - self.yf) > 0.15:
            if self.hold >= 1 and abs(raw - self.prev) <= 0.10:
                self.yf, self.hold = raw, 0
            else:
                self.hold += 1
            self.prev = raw
            return self.yf
        self.hold = 0
        self.yf = ALPHA * raw + (1.0 - ALPHA) * self.yf
        self.prev = raw
        return self.yf

class PICorr:
    def __init__(self, kp, ti, lo, hi):
        self.kp, self.ti, self.lo, self.hi = kp, ti, lo, hi
        self.ki, self.sat = 0.0, 0
    def update(self, e):
        dki = self.kp * (DT / self.ti) * e
        if self.sat * e > 0:
            dki = 0.0
        self.ki += dki
        return float(np.clip(1.0 + self.kp * e + self.ki, self.lo, self.hi))

def constrain(u, up):
    return float(np.clip(np.clip(u, PUMP_MIN, PUMP_MAX), up - RATE_MAX, up + RATE_MAX))

# ---------- 4. 一次辨识：慢回路伪随机激励，最小二乘线性模型 ----------
n_id = 1200
load_id = (2.9 + 1.0 * np.sin(2 * np.pi * np.arange(n_id) / 96.0)
           + 0.4 * np.sin(2 * np.pi * np.arange(n_id) / 31.0)
           + rng.normal(0, 0.03, n_id))
seg = rng.choice([-6.0, -3.0, 3.0, 6.0], size=n_id)
flip = rng.random(n_id) < 0.12
prbs = np.zeros(n_id)
for k in range(1, n_id):
    prbs[k] = seg[k] if flip[k] else prbs[k - 1]
dose_id = np.clip((load_id - TARGET) / K_CHEM + prbs, 8.0,
                  (load_id - 0.15) / K_CHEM)
y_id = np.zeros(n_id) + 0.3
for k in range(1, n_id):
    kd = k - DELAY
    y_id[k] = plant_step(y_id[k - 1], dose_id[kd if kd >= 0 else 0],
                         load_id[kd if kd >= 0 else 0], K_CHEM)
y_id += rng.normal(0, 0.012, n_id)
ks = np.arange(DELAY, n_id - 1)
Xd = np.column_stack([y_id[ks], dose_id[ks + 1 - DELAY],
                      load_id[ks + 1 - DELAY], np.ones_like(ks, dtype=float)])
a_id, b_id, g_id, c_id = np.linalg.lstsq(Xd, y_id[ks + 1], rcond=None)[0]
rmse_id = float(np.sqrt(np.mean((y_id[ks + 1] - Xd @ np.array([a_id, b_id, g_id, c_id])) ** 2)))

# ---------- 5. 枚举式 MPC（慢回路 H=6 拍=90 min，9 候选，DMC 偏差限幅）----------
H, N_CAND = 6, 9
W_SOFT, W_HARD, W_LIN, W_SLUDGE, W_DU = 25.0, 3000.0, 200.0, 0.030, 5.0e-5
G_BIAS, BIAS_CLAMP = 0.20, 0.06   # 残差限幅：超出即疑似仪表问题，不再照单全收

def mpc_cost(u_cand, y_now, j, ux, bias, Qd, TPa, TPfc, fc_known, fc_noise, b_use):
    """fc_known=True（排产预报）时未来负荷取真实预报；否则按持续性假设外推当前值。"""
    yp, jc = y_now, 0.0
    for h in range(1, H + 1):
        jf = j + h - DELAY
        if jf <= j - 1:
            dose_h = Lh_to_mgL(ux[jf], Qd[jf])
            load = TPa[jf]
        else:
            jfc = min(jf, M - 1)
            dose_h = Lh_to_mgL(u_cand, Qd[jfc])
            if fc_known:
                load = TPfc[jfc]
            else:
                # 以当前实测为锚、叠加典型日形状增量（当日实际偏离典型日的部分不可预知）
                k0 = min(max(j - WARM, 0), ND - 1)
                kk = min(max(jfc - WARM, 0), ND - 1)
                shape = TP_pat[kk] - TP_pat[k0]
                load = (TPa[j] + shape) * (1.0 + fc_noise[j])
        yp = max(0.03, a_id * yp + b_use * dose_h + g_id * load + c_id) + bias
        sludge = SLUDGE_COEF * u_cand * C_SOL * RHO
        jc += (W_SOFT * (yp - 0.27) ** 2 + W_HARD * max(0.0, yp - LIMIT) ** 2
               + W_LIN * max(0.0, yp - LIMIT) + W_SLUDGE * sludge) * (DT / 60.0)
    jc += W_DU * (u_cand - ux[j - 1]) ** 2
    return jc

# ---------- 6. 策略回放 ----------
def replay(strategy, q_day, tp_day, k_day, fc_known, drift, cold_model):
    Qd = np.concatenate([np.full(WARM, Q0 * 0.98), q_day])
    TPa = np.concatenate([np.full(WARM, 2.6), tp_day])
    Ke = np.concatenate([np.full(WARM, K_CHEM), k_day])
    # 排产预报：真实未来负荷 ±2%；其余场景持续性外推，当前估计 ±3%
    TPfc = TPa * (1.0 + rng.normal(0, 0.02, TPa.shape)) if fc_known else TPa.copy()
    fc_noise = rng.normal(0, 0.05, M)
    b_use = b_id * 0.8 if cold_model else b_id   # 换季再辨识后的投加系数
    yr, ur = np.zeros(ND), np.zeros(ND)
    ux = np.zeros(M)
    ux[:WARM] = mgL_to_Lh((2.6 - TARGET) / K_CHEM, Q0 * 0.98)
    az = Analyzer()
    pi = PICorr(0.6, 80.0, 0.8, 1.2)
    y = 0.3
    yhat, bias = 0.3, 0.0
    for j in range(WARM, M):
        k = j - WARM
        jd = j - DELAY
        y = plant_step(y, Lh_to_mgL(ux[jd], Qd[jd]), TPa[jd], Ke[jd]) \
            + rng.normal(0, 0.006)
        raw = y + rng.normal(0, 0.025) + drift[k]
        yf = az.read(raw)
        u_ff = mgL_to_Lh(max(0.0, TPa[j] - TARGET) / K_CHEM, Qd[j])
        if strategy == "manual":
            u_ask = 1950.0 if 6.5 <= hours[k] < 10.0 else 1700.0
        elif strategy == "FFPI":
            e = 0.0 if abs(yf - TARGET) < DEADBAND else yf - TARGET
            u_ask = u_ff * pi.update(e)
        else:
            # DMC 偏差校正并限幅：残差超出模型合理范围视为仪表问题，不再照单全收
            yp_open = (a_id * yhat + b_use * Lh_to_mgL(ux[jd], Qd[jd])
                       + g_id * TPa[jd] + c_id)
            bias = float(np.clip(G_BIAS * (yf - yp_open) + (1.0 - G_BIAS) * bias,
                                 -BIAS_CLAMP, BIAS_CLAMP))
            yhat = yp_open + bias
            shock_ahead = fc_known and np.any(TPfc[j + 1:j + H + 1] > 4.0)
            hi_mult = 1.35 if cold_model else 1.20   # 换季再辨识提示效率下降时放宽上限
            if shock_ahead:                          # 预报冲击：放开上限提前布防
                lo = max(PUMP_MIN, ux[j - 1] - RATE_MAX)
                hi = min(PUMP_MAX, ux[j - 1] + RATE_MAX)
            else:
                lo = max(PUMP_MIN, u_ff * 0.85, ux[j - 1] - RATE_MAX)
                hi = min(PUMP_MAX, u_ff * hi_mult, ux[j - 1] + RATE_MAX)
            cands = np.linspace(lo, hi, N_CAND)
            u_ask = cands[int(np.argmin([mpc_cost(c, yf, j, ux, bias, Qd, TPa, TPfc,
                                          fc_known, fc_noise, b_use) for c in cands]))]
        u_ask0 = u_ask
        ux[j] = constrain(u_ask, ux[j - 1])
        pi.sat = 1 if ux[j] < u_ask0 - 1 else (-1 if ux[j] > u_ask0 + 1 else 0)
        yr[k], ur[k] = y, ux[j]
    return yr, ur

# ---------- 7. 指标：药耗 / 超标 / EQI 轻量代理 / 综合得分 ----------
rows = []
for sc in SCENARIOS:
    q_day, tp_day, k_day, fc_known, drift = make_scenario(sc)
    cold = (sc == "低温季")
    for strat, lab in [("manual", "人工经验"), ("FFPI", "前馈+PI"), ("MPC", "枚举MPC")]:
        y, u = replay(strat, q_day, tp_day, k_day, fc_known, drift,
                      cold_model=(cold and strat == "MPC"))
        dry = u.sum() * DT / 60.0 * RHO * C_SOL          # kg 干粉/d
        over_h = float(np.sum(y > LIMIT) * DT / 60.0)
        eqi = float(np.sum(q_day * y * DT / 60.0) / 1000.0)  # kg P/d，EQI 代理
        rows.append([sc, lab, dry, over_h, y.max(), y.mean(), eqi])
df = pd.DataFrame(rows, columns=["场景", "策略", "PAC干粉kg/d", "超标h",
                                 "峰值TP", "平均TP", "EQI代理kgP/d"])

# 综合成本（BSM 的 EQI 与 OCI 必须联看，这里做货币化轻量折算，单价均为示例口径）：
#   药剂 2000 元/t 干粉；化学污泥处置 300 元/tDS（产泥 0.5 kgDS/kg 干粉）；
#   出水磷环境代价 56 元/kgP（按环保税水污染物当量上限示意）；
#   红线超标罚款折算 5000 元/h（仅教学示意，实际按地方执法与环评口径）
PRICE_CHEM, PRICE_SLUDGE, PRICE_P, PRICE_OVER = 2.0, 0.15, 56.0, 5000.0
df["综合成本元/d"] = (PRICE_CHEM * df["PAC干粉kg/d"]
                    + PRICE_SLUDGE * df["PAC干粉kg/d"]
                    + PRICE_P * df["EQI代理kgP/d"]
                    + PRICE_OVER * df["超标h"]).round(0)
print(df.round(3).to_string(index=False))

print("\n=== 六场景合计 ===")
summ = []
for lab in ["人工经验", "前馈+PI", "枚举MPC"]:
    d = df[df["策略"] == lab]
    dm = df[df["策略"] == "人工经验"]
    summ.append([lab, round(d["PAC干粉kg/d"].sum()),
                 round(d["PAC干粉kg/d"].sum() / dm["PAC干粉kg/d"].sum() * 100, 1),
                 round(d["超标h"].sum(), 2), round(d["EQI代理kgP/d"].sum(), 1),
                 round(d["综合成本元/d"].sum())])
print(pd.DataFrame(summ, columns=["策略", "六场景药耗kg", "药耗指数%", "超标h合计",
                                  "EQI kgP合计", "综合成本元合计"]).to_string(index=False))
print("\n慢回路辨识 a=%.4f b=%.5f g=%.4f RMSE=%.4f（理论 a=%.4f b=%.5f g=%.4f）"
      % (a_id, b_id, g_id, rmse_id, A, (1 - A) * (-K_CHEM), 1 - A))

# ---------- 8. 三面板热力图：药耗 / 超标时长 / EQI ----------
labels = ["人工经验", "前馈+PI", "枚举MPC"]
def mat(col):
    return np.array([[df[(df["场景"] == sc) & (df["策略"] == lab)][col].iloc[0]
                      for sc in SCENARIOS] for lab in labels])
Z_chem, Z_over, Z_eqi = mat("PAC干粉kg/d"), mat("超标h"), mat("EQI代理kgP/d")

fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6))
panels = [(Z_chem, "PAC 干粉 kg/d", "YlGnBu_r"),
          (Z_over, "超标时长 h（红线 0.5）", "Reds"),
          (Z_eqi, "EQI 代理 出水磷负荷 kgP/d", "YlOrRd")]
for ax, (Z, ttl, cmap) in zip(axes, panels):
    im = ax.imshow(Z, cmap=cmap, aspect="auto")
    ax.set_xticks(range(len(SCENARIOS)), SCENARIOS, rotation=20, fontsize=9)
    ax.set_yticks(range(len(labels)), labels, fontsize=10)
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            ax.text(j, i, ("%.2f" % Z[i, j]).rstrip("0").rstrip("."),
                    ha="center", va="center", fontsize=9,
                    color="black" if Z[i, j] < Z.max() * 0.62 else "white")
    ax.set_title(ttl, fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
fig.suptitle("策略回放台：六种场景 × 三种加药策略（合成数据，典型参数实算）", fontsize=13)
plt.tight_layout(rect=[0, 0, 1, 0.95])
out = "../images/fig07_04_scenario_heatmap.png"
plt.savefig(out, dpi=130)
print("图已输出", out)
