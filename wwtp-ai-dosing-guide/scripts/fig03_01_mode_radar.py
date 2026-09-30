"""
图 03-1：五种传统加药模式在五个能力维度上的雷达图（定性评分实算）
运行：python3 fig03_01_mode_radar.py
输出：../images/fig03_01_mode_radar.png
说明：1~5 分为作者基于多厂审计经验的定性评分（5 分最好），不是实测统计值；
其中『过量控制』为过量程度的反向指标，分越高代表过量越少、越节约。
"""
import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC"]
plt.rcParams["axes.unicode_minus"] = False

# 五个维度全部约定：分值越高越好
dims = ["响应速度", "投加精度", "过量控制（过量少）", "省人力", "经验可复制性"]

# 五种现场最常见的投加模式，评分见文件头说明
modes = {
    "①固定泵频":          [1, 1, 1, 5, 5],
    "②流量分时段人工表":   [2, 2, 2, 4, 3],
    "③看出水化验日调2~3次": [2, 3, 3, 2, 2],
    "④看在线仪表手动微调":  [4, 3, 3, 1, 2],
    "⑤简易PID投运不整定":  [3, 2, 2, 3, 3],
}

colors = ["#c62828", "#ef6c00", "#f9a825", "#1565c0", "#6a1b9a"]

N = len(dims)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(figsize=(9, 7.5), subplot_kw=dict(polar=True))
for (name, vals), color in zip(modes.items(), colors):
    v = vals + vals[:1]
    ax.plot(angles, v, color=color, lw=2, label=name)
    ax.fill(angles, v, color=color, alpha=0.08)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(dims, fontsize=11)
ax.set_yticks([1, 2, 3, 4, 5])
ax.set_yticklabels(["1", "2", "3", "4", "5"], fontsize=9, color="#555")
ax.set_ylim(0, 5)
ax.set_title("五种传统加药模式能力雷达（1~5 分，定性评分，5 分最好）",
             fontsize=13, pad=22)
ax.legend(loc="upper right", bbox_to_anchor=(1.28, 1.12), fontsize=10, frameon=False)

plt.tight_layout()
out = "../images/fig03_01_mode_radar.png"
plt.savefig(out, dpi=130, bbox_inches="tight")
print("saved:", out)
for name, vals in modes.items():
    print(f"{name}: 总分{sum(vals)}/25  {vals}")
