"""
图 06-3：加药量预测建模完整实战（10 万吨/天市政厂，一年 15 分钟合成数据）
流程：合成数据 -> EDA -> 特征工程 -> 按时间切分 -> 三模型对比 ->
      特征重要性与PDP -> 残差分工况 -> 三策略业务仿真 -> 历史标签陷阱演示
运行：python3 fig06_03_dosing_model.py
输出：../images/fig06_03_pred_scatter.png
      ../images/fig06_03_feature_importance.png
      ../images/fig06_03_strategy_compare.png
说明：全部为合成数据；标签=维持出水TP约0.30 mg/L所需PAC药液流量，
      由化学计量机理（04-1公式）+ 随水温水质漂移的beta + 噪声生成，可复现。
"""
import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC"]
plt.rcParams["axes.unicode_minus"] = False

rng = np.random.default_rng(42)

# ================================================================
# 一、合成一年 15 分钟数据（365 天 x 96 = 35040 行）
# ================================================================
N = 365 * 96
dt = pd.date_range("2025-01-01", periods=N, freq="15min")
doy = dt.dayofyear.to_numpy()
hour = dt.hour.to_numpy() + dt.minute.to_numpy()/60.0

# 水温：1月中旬最冷约8度，7月最热约28度
temp = 18.0 - 10.0*np.cos(2*np.pi*(doy - 15)/365.0)

# 进水流量：日周期（早高峰9点、晚高峰20点）+ AR(1) 扰动
diurnal = 0.10*np.sin(2*np.pi*(hour - 9)/24) + 0.08*np.sin(4*np.pi*(hour - 9)/24)
ar = np.zeros(N)
for i in range(1, N):
    ar[i] = 0.985*ar[i-1] + rng.normal(0, 0.012)
Q = 4167.0*(1.0 + diurnal + ar)

# 降雨：全年约24场，每场2~10小时，带来流量冲击与水质稀释
rain = np.zeros(N)
ev_day = rng.choice(np.arange(20, 350), size=24, replace=False)
for d in ev_day:
    s = int(d*96 + rng.integers(0, 72))
    L = int(rng.integers(8, 40))              # 2~10 小时
    inten = rng.uniform(0.15, 0.55)
    e = min(s+L, N)
    rain[s:e] = inten*np.sin(np.pi*np.arange(e-s)/L)
Q = Q*(1.0 + 0.85*rain)

# 进水 TP：基准4.0，早高夜低，慢变水质，雨天稀释
qual = 0.55*np.sin(2*np.pi*(doy - 60)/365.0)
tp_in = 4.0 + 0.55*np.sin(2*np.pi*(hour - 8)/24) + qual*0.6 \
        + rng.normal(0, 0.12, N) - 0.9*rain
tp_in = np.clip(tp_in, 1.4, 8.0)

# 投加系数 beta：随水温、低负荷、雨天、慢变水质漂移（机理见04-1）
cold = 1.0/(1.0 + np.exp((temp - 13.0)/2.2))          # 低温上浮
low_load = np.clip((4.0 - tp_in)/2.5, 0, 1.0)         # 低浓度OH竞争
qual_ar = np.zeros(N)
for i in range(1, N):
    qual_ar[i] = 0.999*qual_ar[i-1] + rng.normal(0, 0.004)
beta = 2.0 + 0.22*cold + 0.35*low_load + 0.12*(rain > 0) \
       + 0.08*qual_ar + rng.normal(0, 0.015, N)

# 机理标签：维持出水TP=0.30所需的10% PAC药液流量 L/h（04-1公式）
P_kg_h = Q*(tp_in - 0.30)/1000.0                      # 每小时去磷量 kg P/h
pac_dry = P_kg_h*(27.0/31.0)*beta/0.1535              # PAC干粉 kg/h（Al2O3 29%）
dose_req = pac_dry/0.10/1.1                           # 10%药液，密度1.1 kg/L
dose_req = dose_req*(1.0 + rng.normal(0, 0.02, N))    # 过程噪声
dose_req = np.clip(dose_req, 300, 2800)

df = pd.DataFrame({
    "time": dt, "Q": Q, "tp_in": tp_in, "temp": temp,
    "rain": rain, "beta": beta, "dose_req": dose_req,
    "hour": hour, "doy": doy})
print("数据量 %d 行；Q 均值 %.0f m3/h；TP_in 均值 %.2f mg/L；水温 %.1f~%.1f 度"
      % (len(df), Q.mean(), tp_in.mean(), temp.min(), temp.max()))
print("beta 范围 %.2f~%.2f，均值 %.2f" % (beta.min(), beta.max(), beta.mean()))
print("所需药液流量 均值 %.0f L/h，P5 %.0f，P95 %.0f L/h"
      % (dose_req.mean(), np.percentile(dose_req, 5),
         np.percentile(dose_req, 95)))

# ================================================================
# 二、EDA 关键数字（相关性/日周期）
# ================================================================
print("相关系数：dose~TP_in %.2f，dose~Q %.2f，dose~temp %.2f"
      % (np.corrcoef(dose_req, tp_in)[0, 1],
         np.corrcoef(dose_req, Q)[0, 1],
         np.corrcoef(dose_req, temp)[0, 1]))

# ================================================================
# 三、特征工程：滞后、滚动统计、周期编码、工况独热
# ================================================================
d = df.copy()
for lag in [4, 16]:
    d["Q_lag%d" % lag] = d["Q"].shift(lag)
    d["tp_lag%d" % lag] = d["tp_in"].shift(lag)
for w in [4, 16, 96]:
    d["Q_mean%d" % w] = d["Q"].shift(1).rolling(w).mean()
    d["Q_std%d" % w] = d["Q"].shift(1).rolling(w).std()
    d["tp_mean%d" % w] = d["tp_in"].shift(1).rolling(w).mean()
d["hour_sin"] = np.sin(2*np.pi*d["hour"]/24)
d["hour_cos"] = np.cos(2*np.pi*d["hour"]/24)
d["doy_sin"] = np.sin(2*np.pi*d["doy"]/365)
d["doy_cos"] = np.cos(2*np.pi*d["doy"]/365)
d["rain_lvl"] = np.select([d["rain"] == 0, d["rain"] < 0.3],
                          ["dry", "light"], default="heavy")
d = pd.get_dummies(d, columns=["rain_lvl"], prefix="r", drop_first=False)
d = d.dropna().reset_index(drop=True)

feat_cols = [c for c in d.columns if c not in
             ("time", "dose_req", "beta", "doy", "hour", "rain")]
X = d[feat_cols].to_numpy(dtype=float)
y = d["dose_req"].to_numpy(dtype=float)

# ================================================================
# 四、按时间切分 70/15/15（测试段约11~12月，秋冬降温）
# ================================================================
n1, n2 = int(len(d)*0.70), int(len(d)*0.85)
Xtr, Xva, Xte = X[:n1], X[n1:n2], X[n2:]
ytr, yva, yte = y[:n1], y[n1:n2], y[n2:]
scaler = StandardScaler().fit(Xtr)
Xtr_s = scaler.transform(Xtr)
Xte_s = scaler.transform(Xte)
print("切分：训练 %d 行（0~%.0f天），验证 %d，测试 %d（第%.0f~%.0f天）"
      % (n1, n1/96, n2-n1, len(d)-n2, n2/96, len(d)/96))

# ================================================================
# 五、三模型对比
# ================================================================
models = {
    "线性回归": LinearRegression(),
    "随机森林": RandomForestRegressor(n_estimators=300, max_depth=18,
                                      min_samples_leaf=5, n_jobs=-1,
                                      random_state=42),
    "梯度提升": HistGradientBoostingRegressor(max_iter=400, learning_rate=0.06,
                                              max_leaf_nodes=31,
                                              l2_regularization=1.0,
                                              random_state=42),
}
preds = {}
rows = []
for name, mdl in models.items():
    Xfit = Xtr_s if name == "线性回归" else Xtr
    Xp = Xte_s if name == "线性回归" else Xte
    mdl.fit(Xfit, ytr)
    p = mdl.predict(Xp)
    preds[name] = p
    mae = mean_absolute_error(yte, p)
    rmse = np.sqrt(mean_squared_error(yte, p))
    r2 = r2_score(yte, p)
    rows.append([name, mae, rmse, 100*mae/yte.mean(), r2])
res = pd.DataFrame(rows, columns=["模型", "MAE", "RMSE", "相对MAE%", "R2"])
print("\n=== 测试集三模型对比（药液 L/h）===")
for _, r in res.iterrows():
    print("%s：MAE %.1f，RMSE %.1f，相对MAE %.2f%%，R2 %.3f"
          % (r["模型"], r["MAE"], r["RMSE"], r["相对MAE%"], r["R2"]))

# ================================================================
# 六、特征重要性（置换重要性）+ 部分依赖 PDP
# ================================================================
hgb = models["梯度提升"]
_pi_idx = np.linspace(n1, len(d)-1, 2500).astype(int)  # 9~12月均匀抽样
perm = permutation_importance(hgb, X[_pi_idx], y[_pi_idx], n_repeats=3,
                              scoring="neg_root_mean_squared_error",
                              random_state=42, n_jobs=-1)
imp = pd.Series(perm.importances_mean, index=feat_cols).sort_values()
top = imp.tail(10)
print("\n置换重要性 Top5：%s"
      % "、".join("%s(%.2f)" % (k, v) for k, v in top.iloc[::-1].head(5).items()))

# ================================================================
# 七、残差分工况（用最优模型 HGB）
# ================================================================
te = d.iloc[n2:].copy()
te["err"] = preds["梯度提升"] - yte
te["abserr"] = te["err"].abs()
print("\n=== 梯度提升残差分工况 MAE（L/h）===")
print("晴天 MAE %.1f（n=%d），雨天 MAE %.1f（n=%d）"
      % (te.loc[te["rain"] == 0, "abserr"].mean(), (te["rain"] == 0).sum(),
         te.loc[te["rain"] > 0, "abserr"].mean(), (te["rain"] > 0).sum()))
print("白天 MAE %.1f，夜间 MAE %.1f"
      % (te.loc[(te["hour"] >= 8) & (te["hour"] <= 20), "abserr"].mean(),
         te.loc[(te["hour"] < 8) | (te["hour"] > 20), "abserr"].mean()))
print("低温<13度 MAE %.1f（n=%d），常温 MAE %.1f"
      % (te.loc[te["temp"] < 13, "abserr"].mean(),
         (te["temp"] < 13).sum(),
         te.loc[te["temp"] >= 13, "abserr"].mean()))

# ================================================================
# 八、业务仿真：出水TP对投加偏差的响应（简化机理）
# ================================================================
def tp_response(d_app, d_req, noise_sd=0.012):
    ratio = d_app/d_req
    tp = np.where(d_app >= d_req,
                  0.30 - 0.80*(ratio - 1.0),
                  0.30 + 1.50*(1.0 - ratio))
    tp = np.clip(tp, 0.12, 1.0) + rng.normal(0, noise_sd, len(ratio))
    return np.clip(tp, 0.08, 1.2)

# 策略1：模型前馈（8%安全余量，输出物理限幅）
ai = np.clip(preds["梯度提升"]*1.08, 300, 2800)
# 策略2：固定过量（取训练期P90常量，全年不调）
fixed = np.full_like(yte, np.percentile(ytr, 90))
# 策略3：人工阶梯（8小时一块，沿用上一块均值+10%保守量，量化到50 L/h）
block = 32
manual = np.zeros_like(yte)
for b in range(0, len(yte), block):
    # 运行员每8h看一次当前流量水质，按当下需要+15%保守量设定并保持
    val = np.round(yte[b]*1.15/50)*50
    manual[b:b+block] = val

def kpi(name, d_app):
    tp = tp_response(d_app, yte)
    pac_t = (d_app*0.25).sum()*0.10*1.1/1000.0      # 测试段干粉吨数
    return {"策略": name, "干粉t": pac_t,
            "药费万元": pac_t*2000/10000,
            "平均TP": tp.mean(),
            "达标率%": 100*(tp <= 0.30).mean(),
            "超一级A%": 100*(tp > 0.50).mean(),
            "tp": tp}

k_ai = kpi("AI前馈", ai)
k_fx = kpi("固定过量", fixed)
k_mn = kpi("人工阶梯", manual)
print("\n=== 测试段（约55天）三策略业务仿真 ===")
for k in (k_fx, k_mn, k_ai):
    print("%s：干粉 %.1f t，药费 %.2f 万元，平均TP %.3f，准IV达标率 %.1f%%，超一级A %.2f%%"
          % (k["策略"], k["干粉t"], k["药费万元"], k["平均TP"],
             k["达标率%"], k["超一级A%"]))
print("AI 相对固定过量：干粉节省 %.1f%%；相对人工阶梯节省 %.1f%%"
      % (100*(1 - k_ai["干粉t"]/k_fx["干粉t"]),
         100*(1 - k_ai["干粉t"]/k_mn["干粉t"])))

# ================================================================
# 九、标签陷阱：拿历史人工投加量当标签会怎样
# ================================================================
op = np.zeros_like(y)
prev = y[:block].mean()
for b in range(0, len(y), block):
    val = np.round(prev*(1.12 + rng.uniform(0, 0.15))/50)*50
    op[b:b+block] = val
    prev = y[b:b+block].mean()
hgb_op = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.06,
                                       max_leaf_nodes=31,
                                       l2_regularization=1.0,
                                       random_state=42)
hgb_op.fit(Xtr, op[:n1])
p_op = hgb_op.predict(Xte)
print("\n=== 标签陷阱对照（测试段）===")
print("真实需要均值 %.0f L/h；历史人工投加均值 %.0f L/h（过量 %.1f%%）"
      % (yte.mean(), op[n2:].mean(), 100*(op[n2:].mean()/yte.mean() - 1)))
print("用历史人工量训出的模型推荐均值 %.0f L/h（比真实需要多 %.1f%%），对人工行为拟合R2 %.3f"
      % (p_op.mean(), 100*(p_op.mean()/yte.mean() - 1), r2_score(op[n2:], p_op)))
print("用正确机理标签的模型推荐均值 %.0f L/h（比真实需要多 %.1f%%）"
      % (preds["梯度提升"].mean(),
         100*(preds["梯度提升"].mean()/yte.mean() - 1)))

# ================================================================
# 图1：三模型 预测 vs 真实 散点
# ================================================================
fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.2), dpi=130, sharex=True,
                         sharey=True)
lo, hi = yte.min(), yte.max()
for ax, name in zip(axes, ["线性回归", "随机森林", "梯度提升"]):
    p = preds[name]
    ax.scatter(yte[::20], p[::20], s=7, alpha=.35, color="#1565c0")
    ax.plot([lo, hi], [lo, hi], "--", color="#c62828", lw=1.4)
    rr = res[res["模型"] == name].iloc[0]
    ax.set_title("%s\nRMSE %.0f，R2 %.3f" % (name, rr["RMSE"], rr["R2"]),
                 fontsize=11)
    ax.grid(alpha=.3)
    ax.set_xlabel("真实所需药液流量（L/h）", fontsize=9.5)
axes[0].set_ylabel("模型预测（L/h）", fontsize=10)
fig.suptitle("测试集预测 vs 真实：越贴近红虚线越好（15分钟合成数据）",
             fontsize=12.5, fontweight="bold")
plt.tight_layout()
plt.savefig("../images/fig06_03_pred_scatter.png")
plt.close()

# ================================================================
# 图2：置换重要性 Top10 + 两个PDP
# ================================================================
fig = plt.figure(figsize=(11.5, 6.2), dpi=130)
gs = fig.add_gridspec(2, 2, width_ratios=[1.25, 1.0])
axb = fig.add_subplot(gs[:, 0])
axb.barh(top.index, top.values, color="#1565c0", alpha=.8)
axb.set_title("置换重要性 Top10：把该特征打乱后误差涨多少", fontsize=11.5,
              fontweight="bold")
axb.set_xlabel("重要性（RMSE增量，L/h）", fontsize=10)
axb.grid(alpha=.3, axis="x")
ax1 = fig.add_subplot(gs[0, 1])
ax2 = fig.add_subplot(gs[1, 1])
_pdp_idx = np.linspace(0, len(d)-1, 3000).astype(int)
# 注意：sklearn 1.9 的 from_estimator 会在同位置新建轴，标签要设在返回的轴上
disp1 = PartialDependenceDisplay.from_estimator(
    hgb, X[_pdp_idx], features=[feat_cols.index("tp_in")],
    feature_names=feat_cols, ax=ax1,
    line_kw={"color": "#2e7d32", "lw": 2})
ax1.remove()
a1 = disp1.axes_[0][0]
a1.set_title("部分依赖：进水TP越高推荐药量越大", fontsize=10.5)
a1.set_xlabel("进水总磷（mg/L）", fontsize=9.5)
a1.set_ylabel("预测药量偏依赖（L/h）", fontsize=9.5)
a1.grid(alpha=.3)
disp2 = PartialDependenceDisplay.from_estimator(
    hgb, X[_pdp_idx], features=[feat_cols.index("temp")],
    feature_names=feat_cols, ax=ax2,
    line_kw={"color": "#ef6c00", "lw": 2})
ax2.remove()
a2 = disp2.axes_[0][0]
a2.set_title("部分依赖：水温越低推荐药量越大", fontsize=10.5)
a2.set_xlabel("水温（摄氏度）", fontsize=9.5)
a2.set_ylabel("预测药量偏依赖（L/h）", fontsize=9.5)
a2.grid(alpha=.3)
plt.tight_layout()
plt.savefig("../images/fig06_03_feature_importance.png")
plt.close()

# ================================================================
# 图3：三策略对比（上：14天出水TP时序；下：药耗柱状+达标率）
# ================================================================
fig, (axu, axd) = plt.subplots(2, 1, figsize=(11.5, 7.2), dpi=130,
                               gridspec_kw={"height_ratios": [1.25, 1.0]})
z = 14*96
xh = np.arange(z)/96.0
axu.plot(xh, k_fx["tp"][:z], color="#8e24aa", lw=.9, alpha=.8,
         label="固定过量（训练期P90常量）")
axu.plot(xh, k_mn["tp"][:z], color="#00838f", lw=.9, alpha=.8,
         label="人工阶梯（8h调一次+10%）")
axu.plot(xh, k_ai["tp"][:z], color="#2e7d32", lw=.9, alpha=.85,
         label="AI模型前馈（+8%余量）")
axu.axhline(0.30, color="#c62828", ls="--", lw=1.3)
axu.text(0.2, 0.315, "准IV类红线 0.30 mg/L", color="#c62828", fontsize=9.5)
axu.set_title("测试段前14天三策略出水总磷仿真", fontsize=12,
              fontweight="bold")
axu.set_ylabel("出水总磷（mg/L）", fontsize=10)
axu.set_xlabel("测试段内天数", fontsize=10)
axu.legend(fontsize=9.5, loc="upper right")
axu.grid(alpha=.3)

names = ["固定过量", "人工阶梯", "AI前馈"]
ks = [k_fx, k_mn, k_ai]
tons = [k["干粉t"] for k in ks]
colors_b = ["#8e24aa", "#00838f", "#2e7d32"]
bars = axd.bar(names, tons, color=colors_b, alpha=.8, width=.55)
for b, k in zip(bars, ks):
    axd.text(b.get_x()+b.get_width()/2, b.get_height()+0.3,
             "%.1f t\n达标率%.0f%%" % (k["干粉t"], k["达标率%"]),
             ha="center", fontsize=10.5, fontweight="bold")
axd.set_ylabel("测试段 PAC 干粉消耗（t）", fontsize=10)
axd.set_title("药耗与准IV达标率：AI耗药最少且达标率不降",
              fontsize=12, fontweight="bold")
axd.set_ylim(0, max(tons)*1.18)
axd.grid(alpha=.3, axis="y")
plt.tight_layout()
plt.savefig("../images/fig06_03_strategy_compare.png")
plt.close()
print("\nsaved 3 figures: pred_scatter / feature_importance / strategy_compare")
