"""
图 07-1：开环建议 L1 / 监督控制 L2 / 全自动闭环 L3 三种模式的权衡对比
运行：cd scripts && python3 fig07_01_level_tradeoff.py
输出：../images/fig07_01_level_tradeoff.png

左图：三模式在 节药效果 / 实施风险 / 改造工作量 三个维度的 1~5 分专家打分
      （分值来自国内 10 万吨级市政厂项目的典型经验区间，非精确统计值）
右图：三模式典型落地投资与真实节药率区间（条形=中值，误差线=经验区间）
"""
import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC"]
plt.rcParams["axes.unicode_minus"] = False

# ---------- 1. 三维度专家打分（1~5，分数越大表示该维度"程度越强"）----------
levels = ["L1 开环建议", "L2 监督控制", "L3 全自动闭环"]
dims = ["节药效果", "实施风险", "改造工作量"]
score = pd.DataFrame(
    {
        "节药效果":   [2.0, 4.0, 5.0],   # 真实兑现的节药：建议执行率打折 / 设定值闭环 / 全天候闭环
        "实施风险":   [1.0, 3.0, 5.0],   # 责任与失控风险递增
        "改造工作量": [1.5, 3.0, 4.5],   # 硬件改造+联锁+调试工作量
    },
    index=levels,
)

# ---------- 2. 投资与节药率经验区间（10 万吨级市政厂，典型经验值）----------
# 投资：软件接入/SCADA弹窗 / +通讯写点+联锁画面 / +仪表冗余+边缘站+全套联锁测试
inv_mid = np.array([30.0, 85.0, 140.0])          # 万元
inv_lo = np.array([15.0, 50.0, 100.0])
inv_hi = np.array([50.0, 120.0, 200.0])
# 真实节药率：L1 受人工执行率拖累，常只兑现理论值的三到五成
sav_mid = np.array([8.0, 20.0, 25.0])            # %
sav_lo = np.array([3.0, 15.0, 18.0])
sav_hi = np.array([12.0, 25.0, 30.0])

# ---------- 3. 出图 ----------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5.2))

x = np.arange(len(dims))
w = 0.24
colors = ["#4C78A8", "#F58518", "#54A24B"]
for i, lv in enumerate(levels):
    ax1.bar(x + (i - 1) * w, score.loc[lv].values, width=w, label=lv, color=colors[i])
ax1.set_xticks(x)
ax1.set_xticklabels(dims)
ax1.set_ylabel("专家打分（1~5，分值越大程度越强）")
ax1.set_ylim(0, 5.6)
ax1.set_title("三模式三维度权衡打分")
ax1.legend(fontsize=9)
ax1.grid(axis="y", alpha=0.3)

ax2b = ax2.twinx()
b1 = ax2.bar(np.arange(3) - 0.2, inv_mid, width=0.38, color="#4C78A8", alpha=0.85,
             yerr=[inv_mid - inv_lo, inv_hi - inv_mid], capsize=4, label="落地投资（万元）")
b2 = ax2b.bar(np.arange(3) + 0.2, sav_mid, width=0.38, color="#E45756", alpha=0.85,
              yerr=[sav_mid - sav_lo, sav_hi - sav_mid], capsize=4, label="真实节药率（%）")
ax2.set_xticks(np.arange(3))
ax2.set_xticklabels(["L1", "L2", "L3"])
ax2.set_ylabel("落地投资（万元，10 万吨级厂）", color="#4C78A8")
ax2b.set_ylabel("真实节药率（%）", color="#E45756")
ax2.set_title("投资规模与真实节药率区间")
ax2.set_ylim(0, 240)
ax2b.set_ylim(0, 40)
ax2.grid(axis="y", alpha=0.3)
lines = [b1, b2]
ax2.legend(lines, [l.get_label() for l in lines], loc="upper left", fontsize=9)

plt.tight_layout()
out = "../images/fig07_01_level_tradeoff.png"
plt.savefig(out, dpi=130)
print("图已输出", out)

# ---------- 4. 正文引用数字 ----------
print("\n=== 三维度打分 ===")
print(score.to_string())
print("\n=== 投资与节药率经验区间（10 万吨级市政厂）===")
for i, lv in enumerate(levels):
    print(f"{lv}: 投资 {inv_lo[i]:.0f}~{inv_hi[i]:.0f} 万元（中值 {inv_mid[i]:.0f}），"
          f"真实节药率 {sav_lo[i]:.0f}%~{sav_hi[i]:.0f}%（中值 {sav_mid[i]:.0f}%）")
