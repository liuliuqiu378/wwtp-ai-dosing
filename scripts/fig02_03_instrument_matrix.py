"""
图 02-3：在线仪表对比矩阵 —— 测量周期、价格量级、维护工作量
运行：python3 fig02_03_instrument_matrix.py
输出：../images/fig02_03_instrument_matrix.png
说明：数值为国内市政污水厂常见工程经验区间的中值（具体项目以询价为准）；
左图测量周期采用对数横轴，右图气泡大小代表单套价格量级。
"""
import matplotlib
matplotlib.use("Agg")
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC"]
plt.rcParams["axes.unicode_minus"] = False

# name, 周期 min, 价格中值 万元, 维护工作量指数 1~10, 类别
rows = [
    ("电磁流量计", 0.02, 1.5, 2, "秒级过程仪表"),
    ("pH 计",     0.03, 0.5, 3, "秒级过程仪表"),
    ("ORP 计",    0.03, 0.4, 2, "秒级过程仪表"),
    ("DO 溶解氧", 0.05, 1.0, 4, "秒级过程仪表"),
    ("超声波液位", 0.05, 0.8, 2, "秒级过程仪表"),
    ("MLSS 浓度计", 0.1, 3.0, 4, "秒级过程仪表"),
    ("COD 分析仪", 15.0, 15.0, 8, "成分分析仪表"),
    ("氨氮分析仪", 15.0, 18.0, 9, "成分分析仪表"),
    ("总磷分析仪", 20.0, 18.0, 9, "成分分析仪表"),
]
df = pd.DataFrame(rows, columns=["name", "period", "price", "maint", "kind"])
print(df.to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.6))

# 左图：测量周期横向条形（对数轴）
ax = axes[0]
cmap = {"秒级过程仪表": "#1565c0", "成分分析仪表": "#c62828"}
y = np.arange(len(df))
ax.barh(y, df["period"], color=[cmap[k] for k in df["kind"]], alpha=.8)
ax.set_xscale("log")
ax.set_yticks(y)
ax.set_yticklabels(df["name"])
ax.invert_yaxis()
ax.set_xlabel("出一个测量值的周期 min（对数轴）")
ax.set_title("测量周期：秒级 vs 10~30 分钟")
for yi, p in zip(y, df["period"]):
    label = f"{p*60:.0f} 秒" if p < 1 else f"{p:.0f} 分钟"
    ax.text(p * 1.25, yi, label, va="center", fontsize=9)
ax.set_xlim(0.01, 200)
ax.grid(axis="x", alpha=.3, which="both")
for k, v in cmap.items():
    ax.scatter([], [], color=v, label=k)
ax.legend(loc="lower right", fontsize=9)

# 右图：价格 vs 维护工作量，气泡大小=价格
ax = axes[1]
for kind, sub in df.groupby("kind"):
    ax.scatter(sub["maint"], sub["price"], s=sub["price"] * 28 + 40,
               color=cmap[kind], alpha=.55, edgecolor="#333", label=kind)
for _, r in df.iterrows():
    ax.annotate(r["name"], (r["maint"], r["price"]),
                xytext=(7, 2), textcoords="offset points", fontsize=9)
ax.set_yscale("log")
ax.set_xlabel("日常维护工作量指数（1=免维护，10=很操心）")
ax.set_ylabel("单套价格量级 万元（对数轴）")
ax.set_xlim(0.5, 10.5)
ax.set_ylim(0.2, 60)
ax.set_title("越贵越操心：成分分析仪价格与维护量双高")
ax.legend(loc="upper left", fontsize=9)
ax.grid(alpha=.3, which="both")

plt.tight_layout()
plt.savefig("../images/fig02_03_instrument_matrix.png", dpi=130)
print("saved ../images/fig02_03_instrument_matrix.png")
