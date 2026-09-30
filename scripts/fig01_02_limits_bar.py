"""
图 01-2：典型市政进水 / GB18918 一级A / 准IV类 限值分组柱状对比（合成典型值）
运行：python3 fig01_02_limits_bar.py
输出：../images/fig01_02_limits_bar.png
说明：本图为典型量级演示，具体项目以实测和当地环评批复为准。
"""
import matplotlib
matplotlib.use("Agg")
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC"]
plt.rcParams["axes.unicode_minus"] = False

# 高量级指标（碳污染与悬浮物）
hi_labels = ["COD", "BOD5", "SS"]
hi_in = [350.0, 160.0, 200.0]      # 典型市政进水，mg/L
hi_a = [50.0, 10.0, 10.0]          # GB 18918-2002 一级A
hi_iv = [30.0, 6.0, np.nan]        # 准IV参照 GB3838 IV类，地表水无 SS 指标

# 营养盐指标（量级小，单独成图避免被压扁）
lo_labels = ["氨氮", "TN", "TP"]
lo_in = [35.0, 45.0, 4.5]
lo_a = [5.0, 15.0, 0.5]
lo_iv = [1.5, 1.5, 0.3]            # TN 取湖库IV类口径，仅部分地区参照

c_in, c_a, c_iv = "#90a4ae", "#fb8c00", "#43a047"

def draw_group(ax, labels, vin, va, viv, title, ylim):
    x = np.arange(len(labels))
    w = 0.25
    b1 = ax.bar(x - w, vin, w, label="典型进水", color=c_in)
    b2 = ax.bar(x, va, w, label="一级A 限值", color=c_a)
    b3bars = [v if not np.isnan(v) else 0 for v in viv]
    b3 = ax.bar(x + w, b3bars, w, label="准IV 参照", color=c_iv)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("浓度 (mg/L)")
    ax.set_ylim(0, ylim)
    ax.set_title(title)
    ax.grid(axis="y", alpha=.3)
    for bars, vals in [(b1, vin), (b2, va), (b3, viv)]:
        for rect, v in zip(bars, vals):
            if np.isnan(v):
                ax.text(rect.get_x() + rect.get_width() / 2, 8,
                        "地表水\n无此指标", ha="center", va="bottom",
                        fontsize=8, color="#558b2f")
            else:
                ax.text(rect.get_x() + rect.get_width() / 2, v + ylim * 0.015,
                        f"{v:g}", ha="center", va="bottom", fontsize=8.5)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5.3), dpi=130)
draw_group(ax1, hi_labels, hi_in, hi_a, hi_iv,
           "碳污染与悬浮物（量级 0-400）", 400)
draw_group(ax2, lo_labels, lo_in, lo_a, lo_iv,
           "营养盐（量级 0-50，TP 看右轴感受）", 50)
ax2.legend(loc="upper right", fontsize=9)
ax1.legend(loc="upper right", fontsize=9)
fig.suptitle("进厂时有多脏、出厂时要多干净：六个核心指标的三道刻度（mg/L）", fontsize=13)
fig.text(0.5, -0.02,
         "注：一级A 依据 GB 18918-2002；准IV 参照 GB 3838-2002 地表水IV类，"
         "BOD 取 6、TN 取湖库口径 1.5、SS 无对应指标。数据为典型参数实算。",
         ha="center", fontsize=8.5, color="#555")
plt.tight_layout(rect=(0, 0.03, 1, 1))
plt.savefig("../images/fig01_02_limits_bar.png")
print("已保存 ../images/fig01_02_limits_bar.png")

df = pd.DataFrame({
    "指标": hi_labels + lo_labels,
    "典型进水": hi_in + lo_in,
    "一级A": hi_a + lo_a,
    "准IV参照": [30, 6, None, 1.5, 1.5, 0.3],
})
print(df.to_string(index=False))
for name, vin, va in [("COD", 350.0, 50.0), ("氨氮", 35.0, 5.0), ("TP", 4.5, 0.5)]:
    print(f"{name} 从典型进水到一级A 需去除: {(1 - va/vin)*100:.1f}%")
