"""
图 06-4：出水总磷预测与软测量建模（30分钟粒度，180天合成数据）
内容：带水力滞后的出水TP过程 -> 滞后特征+HistGB -> 30/60/120分钟直接多步预测
      对比持久性基线；置换重要性代替SHAP
运行：python3 fig06_04_effluent_tp.py
输出：../images/fig06_04_horizon_error.png
      ../images/fig06_04_forecast_compare.png
说明：合成数据，滞后核均值约170~180分钟（与05-3互相关算例一致），不代表具体水厂。
"""
import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC"]
plt.rcParams["axes.unicode_minus"] = False

rng = np.random.default_rng(42)

# ================ 一、合成180天、30分钟粒度过程数据（8640行）================
N = 180*48
dt = pd.date_range("2025-03-01", periods=N, freq="30min")
doy = dt.dayofyear.to_numpy()
hour = dt.hour + dt.minute/60.0

temp = 18.0 - 10.0*np.cos(2*np.pi*(doy - 15)/365.0)
diurnal = 0.10*np.sin(2*np.pi*(hour - 9)/24) + 0.08*np.sin(4*np.pi*(hour - 9)/24)
ar = np.zeros(N)
for i in range(1, N):
    ar[i] = 0.99*ar[i-1] + rng.normal(0, 0.01)
Q = 4167.0*(1.0 + diurnal + ar)
rain = np.zeros(N)
for d in rng.choice(np.arange(10, 170), size=12, replace=False):
    s = int(d*48 + rng.integers(0, 36)); L = int(rng.integers(6, 24))
    e = min(s+L, N)
    rain[s:e] = rng.uniform(0.2, 0.5)*np.sin(np.pi*np.arange(e-s)/L)
Q = Q*(1.0 + 0.8*rain)
tp_in = np.clip(4.0 + 0.55*np.sin(2*np.pi*(hour-8)/24)
                + rng.normal(0, 0.12, N) - 0.9*rain, 1.4, 8)

# 真实需要药液量（04-1机理，beta随水温漂移）
beta = 2.0 + 0.22/(1.0 + np.exp((temp-13)/2.2)) \
       + 0.35*np.clip((4.0-tp_in)/2.5, 0, 1.0) + 0.12*(rain > 0)
dose_req = Q*(tp_in-0.30)/1000.0*(27/31)*beta/0.1535/0.10/1.1

# 实际投加：2小时阶梯粗调（每4步看一次），带约-5%~+12%执行偏差
dose_app = np.empty(N)
for b in range(0, N, 4):
    bias = rng.uniform(-0.10, 0.15)
    if rng.random() < 0.20:                      # 两成时段没跟上，欠投10%~22%
        bias = -rng.uniform(0.10, 0.22)
    dose_app[b:b+4] = dose_req[b]*(1.0 + bias)

# 出水TP：投加盈亏经水力滞后核（30分钟步长，核峰约第6步=3h）展宽
pressure = 1.5*(1.0 - dose_app/dose_req) + 0.06*(tp_in - 4.0) \
           + 0.004*(13.0 - temp)
lags = np.arange(0, 18)
kernel = np.where(lags >= 3, (lags-3)*np.exp(-(lags-6)/3.5), 0.0)
kernel = kernel/kernel.sum()
tp_out = 0.30 + 2.0*np.convolve(pressure, kernel, mode="full")[:N]
noise = np.zeros(N)
for i in range(1, N):
    noise[i] = 0.55*noise[i-1] + rng.normal(0, 0.008)
tp_out = np.clip(tp_out + noise, 0.08, 1.2)

df = pd.DataFrame({"time": dt, "Q": Q, "tp_in": tp_in, "temp": temp,
                   "rain": rain, "dose": dose_app, "tp_out": tp_out,
                   "hour": hour})
print("数据 %d 行；出水TP 均值 %.3f，P95 %.3f mg/L"
      % (len(df), tp_out.mean(), np.percentile(tp_out, 95)))

# ================= 二、特征：仅用预测时刻之前可得的信息 =================
d = df.copy()
for L in [1, 2, 3, 4, 6, 8]:
    d["tp_lag%d" % L] = d["tp_out"].shift(L)
for w in [4, 8]:
    d["tp_mean%d" % w] = d["tp_out"].shift(1).rolling(w).mean()
    d["tp_std%d" % w] = d["tp_out"].shift(1).rolling(w).std()
for L in [2, 6]:
    d["Q_lag%d" % L] = d["Q"].shift(L)
    d["dose_lag%d" % L] = d["dose"].shift(L)
d["hour_sin"] = np.sin(2*np.pi*d["hour"]/24)
d["hour_cos"] = np.cos(2*np.pi*d["hour"]/24)
d = d.dropna().reset_index(drop=True)

feat = [c for c in d.columns if c not in ("time", "hour", "tp_out")]
X = d[feat].to_numpy(float)
y = d["tp_out"].to_numpy(float)

# ================= 三、按时间切分 70/15/15 =================
n1, n2 = int(len(d)*0.70), int(len(d)*0.85)
Xtr, Xte = X[:n1], X[n2:]
ytr, yte = y[:n1], y[n2:]
print("训练 %d 行，测试 %d 行（第%d~%d天）" % (n1, len(d)-n2, n2/48, len(d)/48))

# ================= 四、30/60/120分钟直接多步预测 =================
horizons = {"30分钟": 1, "60分钟": 2, "120分钟": 4}
preds = {}
print("\n=== 不同预测horizon误差（测试集，mg/L）===")
rows = []
for name, h in horizons.items():
    # 目标：未来第h步的出水TP，特征整体前移h行对齐
    Xh_tr, yh_tr = Xtr[:-h], ytr[h:]
    Xh_te, yh_te = Xte[:-h], yte[h:]
    m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.06,
                                      max_leaf_nodes=31, l2_regularization=1.0,
                                      random_state=42)
    m.fit(Xh_tr, yh_tr)
    p = m.predict(Xh_te)
    persist = yte[:len(yh_te)]          # 持久性：只用预测原点时刻观测y(t)顶h步
    mae, rmse = mean_absolute_error(yh_te, p), np.sqrt(
        mean_squared_error(yh_te, p))
    mae_b = mean_absolute_error(yh_te, persist)
    rmse_b = np.sqrt(mean_squared_error(yh_te, persist))
    preds[name] = (yh_te, p)
    rows.append([name, mae, rmse, mae_b, rmse_b,
                 100*(1-mae/mae_b)])
    print("%s：HistGB MAE %.4f / RMSE %.4f；持久性基线 MAE %.4f / RMSE %.4f；MAE改善 %.1f%%"
          % (name, mae, rmse, mae_b, rmse_b, 100*(1-mae/mae_b)))

# ================= 五、置换重要性（60分钟模型，代替SHAP）=================
h = 2
m60 = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.06,
                                    max_leaf_nodes=31, l2_regularization=1.0,
                                    random_state=42)
m60.fit(Xtr[:-h], ytr[h:])
perm = permutation_importance(m60, Xte[:-h], yte[h:], n_repeats=3,
                              scoring="neg_root_mean_squared_error",
                              random_state=42, n_jobs=-1)
imp = pd.Series(perm.importances_mean, index=feat).sort_values()[::-1]
print("\n60分钟模型置换重要性 Top5（数字为打乱后RMSE增量，mg/L）：%s"
      % "、".join("%s(%.4f)" % (k, v) for k, v in imp.head(5).items()))

# ================= 图1：horizon vs 误差（双子图）=================
rdf = pd.DataFrame(rows, columns=["h", "mae", "rmse", "mae_b", "rmse_b", "imp"])
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.6), dpi=130)
xx = np.arange(len(rdf)); w = 0.36
a1.bar(xx-w/2, rdf["mae_b"], w, label="持久性基线（明天=今天）", color="#90a4ae")
a1.bar(xx+w/2, rdf["mae"], w, label="滞后特征+梯度提升", color="#1565c0")
for i, r in rdf.iterrows():
    a1.text(i+w/2, r["mae"]+0.001, "%.4f" % r["mae"], ha="center",
            fontsize=9, fontweight="bold")
    a1.text(i-w/2, r["mae_b"]+0.001, "%.4f" % r["mae_b"], ha="center",
            fontsize=9, color="#546e7a")
a1.set_xticks(xx); a1.set_xticklabels(rdf["h"])
a1.set_ylabel("MAE（mg/L）"); a1.set_xlabel("预测提前量")
a1.set_title("MAE随horizon上升，模型始终优于基线", fontsize=11.5,
             fontweight="bold")
a1.legend(fontsize=9); a1.grid(alpha=.3, axis="y")
a2.bar(xx-w/2, rdf["rmse_b"], w, label="持久性基线", color="#90a4ae")
a2.bar(xx+w/2, rdf["rmse"], w, label="滞后特征+梯度提升", color="#c62828")
for i, r in rdf.iterrows():
    a2.text(i+w/2, r["rmse"]+0.001, "%.4f" % r["rmse"], ha="center",
            fontsize=9, fontweight="bold")
    a2.text(i-w/2, r["rmse_b"]+0.001, "%.4f" % r["rmse_b"], ha="center",
            fontsize=9, color="#546e7a")
a2.set_xticks(xx); a2.set_xticklabels(rdf["h"])
a2.set_ylabel("RMSE（mg/L）"); a2.set_xlabel("预测提前量")
a2.set_title("RMSE同样优于基线（大错更少）", fontsize=11.5, fontweight="bold")
a2.legend(fontsize=9); a2.grid(alpha=.3, axis="y")
plt.tight_layout()
plt.savefig("../images/fig06_04_horizon_error.png")
plt.close()

# ================= 图2：测试段10天预测时序对比 =================
fig, ax = plt.subplots(figsize=(11.5, 5.2), dpi=130)
z = 10*48
xa = np.arange(z)/48.0
ax.plot(xa, preds["30分钟"][0][:z], color="#263238", lw=1.4,
        label="实际出水TP（合成）")
ax.plot(xa, preds["30分钟"][1][:z], color="#1565c0", lw=1.0, ls="--",
        label="提前30分钟预测")
ax.plot(xa, preds["60分钟"][1][:z], color="#2e7d32", lw=1.0, ls="--",
        label="提前60分钟预测")
ax.plot(xa, preds["120分钟"][1][:z], color="#ef6c00", lw=1.0, ls="--",
        label="提前120分钟预测")
ax.axhline(0.30, color="#c62828", ls=":", lw=1.3)
ax.text(0.1, 0.315, "准IV类 0.30 mg/L", color="#c62828", fontsize=9.5)
ax.set_title("测试段前10天：提前越久曲线越钝，但拐点方向仍在",
             fontsize=12.5, fontweight="bold")
ax.set_xlabel("测试段内天数"); ax.set_ylabel("出水总磷（mg/L）")
ax.legend(fontsize=9.5, ncol=2); ax.grid(alpha=.3)
plt.tight_layout()
plt.savefig("../images/fig06_04_forecast_compare.png")
plt.close()
print("\nsaved: fig06_04_horizon_error.png / fig06_04_forecast_compare.png")
