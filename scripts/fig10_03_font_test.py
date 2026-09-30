import matplotlib
matplotlib.use("Agg")
import sys, platform
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC"]
plt.rcParams["axes.unicode_minus"] = False

print("Python:", platform.python_version(), "| 系统:", sys.platform)
print("numpy", np.__version__, "| pandas", pd.__version__,
      "| matplotlib", matplotlib.__version__)

rng = np.random.default_rng(42)
days = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
dose = np.array([260, 255, 270, 310, 290, 240, 230]) + rng.normal(0, 8, 7)

fig, ax = plt.subplots(figsize=(9, 5), dpi=130)
ax.bar(days, dose, color="#4c78a8")
ax.set_title("字体测试：一周 PAC 投加量（合成示例数据）")
ax.set_xlabel("日期")
ax.set_ylabel("投加量 kg/d")
for x, y in zip(days, dose):
    ax.text(x, y + 2, f"{y:.0f}", ha="center", va="bottom", fontsize=9)
ax.set_ylim(0, max(dose) * 1.15)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("../images/fig10_03_font_test.png")
print("中文测试图已生成：images/fig10_03_font_test.png")
print("周均投加量:", round(float(np.mean(dose)), 1), "kg/d")
