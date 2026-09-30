"""
图 06-5：异常检测与工况识别（14天、15分钟多通道合成数据）
注入三类异常：仪表卡死、单点尖峰、真实进水冲击
检测：规则（卡死滚动极差 / Hampel / 雨量+流量复合冲击规则）+ IsolationForest
输出混淆矩阵与分类召回，时序图标注异常与报警
运行：python3 fig06_05_anomaly.py
输出：../images/fig06_05_anomaly.png
说明：合成数据，阈值为演示取值，现场需按本厂数据质量分重新标定。
"""
import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC"]
plt.rcParams["axes.unicode_minus"] = False

rng = np.random.default_rng(42)

# ================= 一、14天正常工况基底 =================
N = 14*96
t = pd.date_range("2025-06-02", periods=N, freq="15min")
hour = t.hour.to_numpy() + t.minute.to_numpy()/60
Q = 4167.0*(1.0 + 0.10*np.sin(2*np.pi*(hour-9)/24)
            + 0.08*np.sin(4*np.pi*(hour-9)/24)) \
    + rng.normal(0, 60, N)
tp = 4.0 + 0.55*np.sin(2*np.pi*(hour-8)/24) + rng.normal(0, 0.10, N)
rain = np.zeros(N)

# 真值标记：normal / stuck / spike / shock
kind = np.array(["normal"]*N, dtype=object)

# 真实冲击：3场暴雨，Q升35%~60%、TP抬1.5~2.5，持续1~3小时
shock_starts = [int(2.2*96), int(6.4*96+30), int(10.8*96)]
for s0 in shock_starts:
    L = int(rng.integers(4, 13))
    e = min(s0+L, N)
    rain[s0:e] = rng.uniform(0.25, 0.55)*np.sin(np.pi*np.arange(e-s0)/L)
    Q[s0:e] *= (1.0 + rng.uniform(0.35, 0.60))
    tp[s0:e] += rng.uniform(1.5, 2.5)*np.sin(np.pi*np.arange(e-s0)/L)
    kind[s0:e] = "shock"

# 仪表卡死：2段，TP表读数冻结2~3小时
for s0, L in [(int(4.0*96+20), 8), (int(9.3*96+40), 12)]:
    tp[s0:s0+L] = tp[s0-1]
    kind[s0:s0+L] = "stuck"

# 单点尖峰：8个毛刺（与冲击时段错开）
spike_idx = []
cand = rng.choice(np.arange(20, N-20), size=8, replace=False)
for s0 in cand:
    if kind[max(0,s0-6):min(N,s0+6)].sum() if False else not (
            "shock" in kind[max(0,s0-6):min(N,s0+6)]
            or "stuck" in kind[max(0,s0-6):min(N,s0+6)]):
        tp[s0] += rng.choice([-1, 1])*rng.uniform(1.0, 1.8)
        kind[s0] = "spike"
        spike_idx.append(s0)

df = pd.DataFrame({"time": t, "Q": Q, "tp": tp, "rain": rain, "kind": kind})
n_shock = int((kind == "shock").sum())
n_stuck = int((kind == "stuck").sum())
n_spike = int((kind == "spike").sum())
print("注入：冲击 %d 点（3场，%dh），卡死 %d 点（2段，%dh），尖峰 %d 点"
      % (n_shock, n_shock*15/60, n_stuck, n_stuck*15/60, n_spike))

# ================= 二、规则检测 =================
s = pd.Series(tp)
# 卡死：在线（trailing）连续4点极差<0.01，约30分钟确认，符合05-2口径
roll_min = s.rolling(4).min()
roll_max = s.rolling(4).max()
stuck_alarm = ((roll_max - roll_min) < 0.01).fillna(False).to_numpy()

med = s.rolling(9, center=True).median()
mad = (s - med).abs().rolling(9, center=True).median()
dev = (s - med).abs()
# 尖峰：同时满足5倍MAD与0.4 mg/L绝对门槛（防止低波动段误报），且非雨天
spike_alarm = np.array(((dev > 5.0*mad) & (dev > 0.40)).fillna(False))
spike_alarm = spike_alarm & (rain == 0)

qmed = pd.Series(Q).rolling(96, min_periods=20).median()
qstd = pd.Series(Q).rolling(96, min_periods=20).std()
shock_rule = np.array(((Q > qmed + 2.5*qstd) & (rain > 0.05)).fillna(False))
# 邻域30分钟内规则任一命中即整段预警，但报警点必须仍有雨量支撑
shock_ext = pd.Series(shock_rule).rolling(3, center=True).max().fillna(
    0).to_numpy().astype(bool)
shock_alarm = shock_ext & (rain > 0.05)

# 仪表类报警先排除真实冲击
inst_alarm = np.array((stuck_alarm | spike_alarm) & (~shock_alarm))

# ================= 三、IsolationForest（前7天无异常段训练）=================
train_end = 7*96
feat = np.column_stack([
    (Q - Q[:train_end].mean())/Q[:train_end].std(),
    (tp - tp[:train_end].mean())/tp[:train_end].std(),
    rain*3.0])
iso = IsolationForest(n_estimators=200, contamination=0.02,
                      random_state=42).fit(feat[:train_end])
if_alarm = iso.predict(feat) == -1

# 融合：工艺异常 = 复合规则 或 雨量背景下IF命中
proc_alarm = shock_alarm | (if_alarm & (rain > 0.05))
any_alarm = inst_alarm | proc_alarm
any_true = kind != "normal"

# ================= 四、评估 =================
tp_p = int((any_alarm & any_true).sum())
fp = int((any_alarm & ~any_true).sum())
fn = int((~any_alarm & any_true).sum())
precision = tp_p/(tp_p+fp)
recall = tp_p/(tp_p+fn)
print("\n=== 综合检测（点级，15分钟一个点）===")
print("TP %d，误报 %d，漏报 %d；精确率 %.1f%%，召回率 %.1f%%"
      % (tp_p, fp, fn, 100*precision, 100*recall))


def recall_type(name, mask_alarm):
    idx = kind == name
    hit = int((mask_alarm & idx).sum())
    print("%s召回：%d/%d = %.0f%%" % (name, hit, idx.sum(),
                                      100*hit/idx.sum()))


print()
recall_type("stuck", inst_alarm)
recall_type("spike", inst_alarm)
recall_type("shock", proc_alarm)
print("正常点误报率 %.2f%%（%d/%d）"
      % (100*fp/(kind == "normal").sum(), fp, (kind == "normal").sum()))

# ================= 五、画图 =================
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11.5, 7), dpi=130, sharex=True)
xh = np.arange(N)/96.0

def shade(ax):
    for i in range(N):
        if kind[i] == "shock":
            c = "#ffcdd2"
        elif kind[i] == "stuck":
            c = "#e1bee7"
        elif kind[i] == "spike":
            c = "#ffe0b2"
        else:
            continue
        ax.axvspan(xh[i], xh[i]+0.25/2, color=c, alpha=.6, zorder=0)

shade(ax1); shade(ax2)
ax1.plot(xh, Q, color="#1565c0", lw=1.0, label="进水流量")
ax1.scatter(xh[proc_alarm], Q[proc_alarm], s=14, marker="v",
            color="#c62828", label="工艺冲击预警", zorder=5)
ax1.set_ylabel("进水流量（m3/h）", fontsize=10)
ax1.set_title("上图：进水流量与复合冲击预警（红区=真实冲击）",
              fontsize=12, fontweight="bold")
ax1.legend(fontsize=9.5, loc="upper right"); ax1.grid(alpha=.3)

ax2.plot(xh, tp, color="#2e7d32", lw=1.0, label="进水总磷")
ax2.scatter(xh[stuck_alarm & ~shock_alarm], tp[stuck_alarm & ~shock_alarm],
            s=16, marker="s", color="#6a1b9a", label="卡死报警", zorder=5)
ax2.scatter(xh[spike_alarm], tp[spike_alarm], s=30, marker="x",
            color="#ef6c00", label="尖峰报警", zorder=5)
ax2.set_ylabel("进水总磷（mg/L）", fontsize=10)
ax2.set_xlabel("天数", fontsize=10)
ax2.set_title("下图：紫区卡死、橙点尖峰——仪表异常与工艺异常分开报",
              fontsize=12, fontweight="bold")
ax2.legend(fontsize=9.5, ncol=3, loc="upper right"); ax2.grid(alpha=.3)
plt.tight_layout()
plt.savefig("../images/fig06_05_anomaly.png")
plt.close()
print("\nsaved: ../images/fig06_05_anomaly.png")
