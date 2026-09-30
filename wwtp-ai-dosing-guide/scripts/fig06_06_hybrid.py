"""
图 06-6：机理+数据灰箱混合模型（一年15分钟合成数据，beta随水温水质慢变）
对比：纯机理（固定beta=2.0）/ 纯ML（HistGB直接预测量）/ 灰箱（机理基础量+ML残差，护栏±15%）
跨季节压力测试：用4~10月训练，11~3月（冬季）测试；另给数据量学习曲线
运行：python3 fig06_06_hybrid.py
输出：../images/fig06_06_hybrid_compare.png
说明：合成数据，不代表具体水厂；护栏口径与回退策略见06-6正文。
"""
import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingRegressor

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC"]
plt.rcParams["axes.unicode_minus"] = False

rng = np.random.default_rng(42)

# =============== 合成一年数据，时序从4月起（末尾即冬季测试段）===============
N = 365*96
dt = pd.date_range("2025-04-01", periods=N, freq="15min")
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

# 慢变水质因子（均值回归OU过程，约8天尺度，长期均值回到0）
qual = np.zeros(N)
for i in range(1, N):
    qual[i] = 0.995*qual[i-1] + rng.normal(0, 0.05)
qual = qual/qual.std()

cold = 1.0/(1.0 + np.exp((temp - 14.0)/2.0))
low_load = np.clip((4.0 - tp_in)/2.5, 0, 1.0)
beta_true = 2.0 + 0.25*cold + 0.12*qual + 0.30*low_load + rng.normal(0, 0.012, N)
# 水质因子的可观测代理（进水碱度/pH综合特征，带测量噪声，现场可得）
qobs = qual + rng.normal(0, 0.15, N)
# 仅在冬季测试段注入初雨冲击：高流量+TP冲到6~8，暖季训练从未见过
storm_n = 0
test_start = (365-151)*96
for d0 in rng.choice(np.arange(220, 360, 7), size=10, replace=False):
    s0 = int(d0*96 + rng.integers(0, 80)); L = int(rng.integers(8, 16))
    if s0 < test_start:
        continue
    e = min(s0+L, N)
    Q[s0:e] *= rng.uniform(1.5, 1.8)
    tp_in[s0:e] = rng.uniform(6.0, 8.0)
    storm_n += 1
print("冬季测试段注入初雨冲击 %d 场" % storm_n)
# 冲击期SS高，beta在低温基础上再上浮约0.10
beta_true = np.where(Q > 4167*1.35,
                     np.maximum(beta_true, 2.1 + 0.25*cold),
                     beta_true)

def dose_of(beta):
    P = Q*(tp_in - 0.30)/1000.0
    return P*(27.0/31.0)*beta/0.1535/0.10/1.1          # 10% PAC药液 L/h

dose_req = dose_of(beta_true)
dose_mech = dose_of(np.full(N, 2.0))                  # 纯机理：固定beta=2.0

# =============== 特征工程（同06-3套路，精简版）===============
dd = pd.DataFrame({"Q": Q, "tp": tp_in, "temp": temp,
                   "qobs": qobs,
                   "dose_req": dose_req, "dose_mech": dose_mech,
                   "hour": hour, "doy": doy})
for L in [4, 16]:
    dd["Q_lag%d" % L] = dd["Q"].shift(L)
    dd["tp_lag%d" % L] = dd["tp"].shift(L)
for w in [16, 96]:
    dd["Q_mean%d" % w] = dd["Q"].shift(1).rolling(w).mean()
    dd["tp_mean%d" % w] = dd["tp"].shift(1).rolling(w).mean()
dd["hour_sin"] = np.sin(2*np.pi*dd["hour"]/24)
dd["hour_cos"] = np.cos(2*np.pi*dd["hour"]/24)
# 注意：不放doy周期特征——冬季doy在训练范围外，树模型会被引到错误季节的叶子
dd = dd.dropna().reset_index(drop=True)

feat = [c for c in dd.columns if c not in
        ("dose_req", "dose_mech", "hour", "doy")]
X = dd[feat].to_numpy(float)
y = dd["dose_req"].to_numpy()
ym = dd["dose_mech"].to_numpy()

# 训练：4~10月（214天）；冬季测试：最后151天
n_tr = 214*96
tr = np.arange(len(dd)) < n_tr
te = np.arange(len(dd)) >= (365-151)*96
bb = beta_true[96:]          # dropna 丢掉前96行，统计量按特征表对齐
tt = temp[96:]
print("训练 %d 行（4~10月，水温 %.0f~%.0f 度），冬季测试 %d 行（水温 %.0f~%.0f 度）"
      % (tr.sum(), tt[tr].min(), tt[tr].max(),
         te.sum(), tt[te].min(), tt[te].max()))
print("冬季 beta_true 均值 %.3f（机理仍按2.0）" % bb[te].mean())

def make_hgb():
    return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.06,
                                         max_leaf_nodes=31,
                                         l2_regularization=1.0,
                                         random_state=42)

# 纯ML：直接学剂量
ml = make_hgb().fit(X[tr], y[tr])
# 灰箱：ML学残差
resid = y - ym
hres = make_hgb().fit(X[tr], resid[tr])

def evaluate(pred, tag):
    err = pred[te] - y[te]
    mae = np.abs(err).mean()
    rel = 100*mae/y[te].mean()
    over = 100*(np.abs(err) > 0.15*y[te]).mean()
    under = 100*(err < -0.05*y[te]).mean()
    bias = 100*err.mean()/y[te].mean()
    print("%s：MAE %.0f L/h，相对MAE %.1f%%，平均偏差 %+.1f%%，欠投超5%%占 %.1f%%，越界率 %.1f%%"
          % (tag, mae, rel, bias, under, over))
    return rel, over

print("\n=== 冬季跨季节测试 ===")
p_mech = ym
p_ml = ml.predict(X)
p_hyb_raw = ym + hres.predict(X)
guard = 0.15*ym                                      # 灰箱护栏：机理±15%
p_hyb = np.clip(p_hyb_raw, ym - guard, ym + guard)
rel_mech, ov_mech = evaluate(p_mech, "纯机理固定beta")
rel_ml, ov_ml = evaluate(p_ml, "纯ML直接预测  ")
rel_hyb, ov_hyb = evaluate(p_hyb, "灰箱机理+残差 ")

# =============== 数据量学习曲线：1/2/3/5/7个月 ===============
months_pts = [30, 60, 90, 150, 214]
lc_ml, lc_hyb = [], []
for days in months_pts:
    k = days*96
    m1 = make_hgb().fit(X[:k], y[:k])
    m2 = make_hgb().fit(X[:k], resid[:k])
    pm = m1.predict(X)
    ph = np.clip(ym + m2.predict(X), ym - guard, ym + guard)
    lc_ml.append(100*np.abs(pm[te] - y[te]).mean()/y[te].mean())
    lc_hyb.append(100*np.abs(ph[te] - y[te]).mean()/y[te].mean())
print("\n学习曲线（冬季测试相对 MAE 百分比）：月数 %s"
      % [1, 2, 3, 5, 7])
print("纯ML：%s" % [round(float(v), 1) for v in lc_ml])
print("灰箱：%s" % [round(float(v), 1) for v in lc_hyb])
print("纯机理水平线：%.1f%%" % rel_mech)

# =============== 画图 ===============
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.8), dpi=130)
names = ["纯机理\n固定beta=2.0", "纯ML\nHistGB", "灰箱\n机理+ML残差"]
rels = [rel_mech, rel_ml, rel_hyb]
ovs = [ov_mech, ov_ml, ov_hyb]
colors_ = ["#90a4ae", "#c62828", "#2e7d32"]
xx = np.arange(3); w = 0.36
b1 = a1.bar(xx-w/2, rels, w, color=colors_, label="相对MAE")
b2 = a1.bar(xx+w/2, ovs, w, color=colors_, alpha=.45, hatch="//",
            label="越界率 偏差大于15%")
for i in range(3):
    a1.text(i-w/2, rels[i]+0.2, "%.1f%%" % rels[i], ha="center",
            fontsize=9.5, fontweight="bold")
    a1.text(i+w/2, ovs[i]+0.2, "%.1f%%" % ovs[i], ha="center", fontsize=9.5)
a1.set_xticks(xx); a1.set_xticklabels(names, fontsize=9.5)
a1.set_ylabel("占真实需要药量的百分比")
a1.set_title("冬季跨季节测试：灰箱误差与越界率双低", fontsize=11.5,
             fontweight="bold")
a1.legend(fontsize=9); a1.grid(alpha=.3, axis="y")

a2.plot([1, 2, 3, 5, 7], lc_ml, "o-", color="#c62828", lw=2,
        label="纯ML直接预测")
a2.plot([1, 2, 3, 5, 7], lc_hyb, "s-", color="#2e7d32", lw=2,
        label="灰箱机理+残差")
a2.axhline(rel_mech, color="#607d8b", ls="--", lw=1.4,
           label="纯机理（无需训练数据）")
a2.set_xlabel("暖季训练数据量（月）"); a2.set_ylabel("冬季测试相对MAE（%）")
a2.set_title("数据量学习曲线：灰箱小样本也稳", fontsize=11.5,
             fontweight="bold")
a2.legend(fontsize=9); a2.grid(alpha=.3)
plt.tight_layout()
plt.savefig("../images/fig06_06_hybrid_compare.png")
plt.close()
print("\nsaved: ../images/fig06_06_hybrid_compare.png")
