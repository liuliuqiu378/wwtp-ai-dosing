"""
图 08-2：边缘时序库年数据量测算（不同点数 × 采集频率）
运行：python3 fig08_02_data_volume.py
输出：../images/fig08_02_data_volume.png
口径：每条记录约 30 字节（时间戳 8B + 标签与质量码约 12B + 数值 8B 及索引摊薄）；
时序库列式压缩后约按原始体积的 1/6 估算（InfluxDB/TDengine 常见经验区间 1/4~1/8）。
"""
import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC"]
plt.rcParams["axes.unicode_minus"] = False

MIN_PER_YEAR = 365 * 24 * 60  # 525600
points = np.array([50, 100, 150, 300, 500, 1000])
intervals = {"10 秒": 1 / 6, "30 秒": 0.5, "1 分钟": 1, "5 分钟": 5}
colors = {"10 秒": "#b71c1c", "30 秒": "#ef6c00", "1 分钟": "#1565c0", "5 分钟": "#2e7d32"}

fig, axes = plt.subplots(1, 2, figsize=(11.8, 5.2), sharex=True)

for label, step_min in intervals.items():
    records = points * MIN_PER_YEAR / step_min
    raw_gb = records * 30 / 1024**3
    comp_gb = raw_gb / 6
    axes[0].plot(points, raw_gb, marker="o", lw=2, color=colors[label], label=label)
    axes[1].plot(points, comp_gb, marker="s", lw=2, color=colors[label], label=label)
    for p, r, c in zip(points, raw_gb, comp_gb):
        if p in (150, 500):
            axes[0].annotate(f"{r:.2f}", (p, r), textcoords="offset points",
                             xytext=(0, 6), ha="center", fontsize=8, color=colors[label])
            axes[1].annotate(f"{c:.2f}", (p, c), textcoords="offset points",
                             xytext=(0, 6), ha="center", fontsize=8, color=colors[label])

for ax, title in zip(axes, ["原始写入体积（约 30 字节/条）", "时序库压缩后落盘（约 1/6 压缩率）"]):
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(points)
    ax.set_xticklabels([str(p) for p in points])
    ax.set_xlabel("采集点位数（个）")
    ax.set_ylabel("年数据量（GB，对数坐标）")
    ax.set_title(title, fontsize=11.5)
    ax.legend(title="采集间隔", fontsize=9, frameon=False)
    ax.grid(alpha=.3, which="both")

fig.suptitle("厂内边缘时序库年数据量测算（数据为典型参数实算，具体以字段设计与压缩率为准）", fontsize=12.5)
plt.tight_layout()
out = "../images/fig08_02_data_volume.png"
plt.savefig(out, dpi=130)
print("saved:", out)
for tag, n in [("小厂典型", 80), ("任务书口径", 150), ("中厂典型", 300), ("大厂典型", 800)]:
    rec = n * MIN_PER_YEAR
    print(f"{tag} {n}点 x 1分钟: 年记录 {rec/1e6:.1f} 百万条, "
          f"原始 {rec*30/1024**3:.2f} GB, 压缩后 {rec*30/1024**3/6:.2f} GB")
