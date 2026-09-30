"""
图 08-4：三个典型厂（A 2万 CASS / B 10万 AAO / C 20万 MBR）投资、年节省与回收期对比
运行：python3 fig08_04_roi_compare.py
输出：../images/fig08_04_roi_compare.png
口径：价格与节药率均取教程统一口径区间内的合成案例值（具体项目以实测与实际报价为准）；
B 厂基线沿用 03-2 逐月台账（PAC 225.2 万、乙酸钠 418.0 万、PAM 28.8 万、次氯酸钠 134.9 万）。
"""
import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC"]
plt.rcParams["axes.unicode_minus"] = False

# ---------- 基线药耗（万元/年；t 为年用量） ----------
# A 厂 2 万 m3/d CASS，年水量 730 万 m3，园区+生活混合水
A_base = {"PAC_t": 292.0, "PAC_price": 0.20,     # 万元/t
          "C_t": 255.5, "C_price": 0.32,
          "PAM_t": 3.65, "PAM_price": 2.0,
          "Cl_t": 292.0, "Cl_price": 0.09}
# B 厂 10 万 m3/d AAO，年水量 3760 万 m3，沿用 03-2 口径
B_base = {"PAC_t": 1250.9, "PAC_price": 0.18,
          "C_t": 1393.3, "C_price": 0.30,
          "PAM_t": 14.4, "PAM_price": 2.0,
          "Cl_t": 1499.3, "Cl_price": 0.09}
# C 厂 20 万 m3/d MBR，年水量 7300 万 m3，准IV类
C_base = {"PAC_t": 2044.0, "PAC_price": 0.19,
          "C_t": 2555.0, "C_price": 0.31,
          "PAM_t": 29.2, "PAM_price": 2.2,
          "Cl_t": 3285.0, "Cl_price": 0.09}

def base_cost(b):
    return {"PAC": b["PAC_t"] * b["PAC_price"], "C": b["C_t"] * b["C_price"],
            "PAM": b["PAM_t"] * b["PAM_price"], "Cl": b["Cl_t"] * b["Cl_price"]}

# ---------- BOM：（名称，数量，单价万元，小计万元），按硬件施工/软件/实施/运维分列 ----------
A_bom = {
 "硬件与施工": [
    ("边缘计算盒子 4核8G256G 含采集与容器运行", 1, 1.20),
    ("在线总磷分析仪 二沉后 国产", 1, 5.80),
    ("加药管路电磁流量计", 2, 0.85),
    ("计量泵远程调节改造 通讯与电动冲程", 3, 0.40),
    ("交换机 网关辅件 小型UPS 一批", 1, 0.80),
    ("线缆管材阀门辅材 一批", 1, 1.80),
    ("施工安装 电气 仪表 就位", 1, 2.60)],
 "软件": [("轻量AI加药软件 除磷闭环加碳源建议加Web看板 三年授权", 1, 14.00)],
 "实施": [("数据审计 建模 影子运行 培训陪产", 1, 7.50)],
 "年运维": [("软件维保 模型迭代 远程支持 每年", 1, 3.60)]}

B_bom = {
 "硬件与施工": [
    ("边缘服务器 至强6核 64G 双固态RAID", 1, 3.80),
    ("边缘采集网关 双网口隔离 4G备份", 2, 0.65),
    ("工业环网交换机", 2, 0.70),
    ("工业防火墙", 1, 2.20),
    ("机柜与UPS配电 一套", 1, 2.20),
    ("硝态氮在线分析仪 缺氧池末端", 1, 9.50),
    ("磷酸盐在线分析仪 好氧末端", 1, 8.50),
    ("加药点电磁流量计", 5, 0.85),
    ("计量泵远程调节改造", 8, 0.42),
    ("线缆桥架管材阀门辅材 一批", 1, 6.50),
    ("施工安装 电气 仪表 开孔就位", 1, 8.00)],
 "软件": [("AI加药软件平台 PAC闭环加碳源前馈加软测量加本地看板", 1, 48.00)],
 "实施": [("数据审计 基线 建模 影子 联调 培训 陪产", 1, 26.00)],
 "年运维": [("软件维保 模型迭代 仪表巡检包 每年", 1, 12.50)]}

C_bom = {
 "硬件与施工": [
    ("边缘服务器 双机热备", 2, 4.50),
    ("边缘采集网关", 4, 0.65),
    ("核心交换机 工业防火墙 单向光闸 一批", 1, 8.50),
    ("机柜与UPS 两套", 2, 2.20),
    ("硝态氮在线分析仪", 2, 9.50),
    ("磷酸盐在线分析仪", 2, 8.50),
    ("DO与ORP在线仪表升级", 4, 1.20),
    ("加药与回流电磁流量计", 8, 0.85),
    ("计量泵远程调节改造", 10, 0.42),
    ("曝气调节阀与鼓风机联控改造", 6, 1.50),
    ("膜系统与脱水机通讯对接", 1, 4.00),
    ("线缆桥架辅材 一批", 1, 9.00),
    ("施工安装 多专业交叉", 1, 12.00)],
 "软件": [
    ("边缘AI软件 加药加曝气协同加脱水优化", 1, 78.00),
    ("数字孪生与仿真验证平台", 1, 45.00),
    ("集团SaaS多厂对标看板 首厂接入一年", 1, 18.00)],
 "实施": [("多变量建模 全流程联调 影子 灰度 全员培训", 1, 48.00)],
 "年运维": [("软件维保 模型迭代 孪生场景更新 SaaS年费", 1, 28.00)]}

def bom_total(bom, categories):
    return {cat: round(sum(q * p for _, q, p in bom[cat]), 2) for cat in categories}

cats = ["硬件与施工", "软件", "实施", "年运维"]
plants = {"A厂 2万 CASS": (A_base, A_bom),
          "B厂 10万 AAO": (B_base, B_bom),
          "C厂 20万 MBR": (C_base, C_bom)}

# ---------- 年节省测算 ----------
# 化学泥饼经验系数：少投 1 t 液体除磷剂约少产 1.8 t 含水率80%泥饼（03-2 口径）
SLUDGE_RATIO = 1.8
results = {}
for name, (b, bom) in plants.items():
    cost = base_cost(b)
    if name.startswith("A"):
        pac_rate, c_rate, pam_rate, sludge_price, power_save = 0.18, 0.12, 0.06, 300, 0.0
    elif name.startswith("B"):
        pac_rate, c_rate, pam_rate, sludge_price, power_save = 0.18, 0.12, 0.07, 350, 0.0
    else:
        pac_rate, c_rate, pam_rate, sludge_price, power_save = 0.20, 0.16, 0.10, 350, 60.0
    pac_save = cost["PAC"] * pac_rate
    c_save = cost["C"] * c_rate
    pam_save = cost["PAM"] * pam_rate
    pac_cut_t = b["PAC_t"] * pac_rate
    sludge_cut_t = pac_cut_t * SLUDGE_RATIO
    sludge_save = sludge_cut_t * sludge_price / 10000.0
    total_save = pac_save + c_save + pam_save + sludge_save + power_save
    sub = bom_total(bom, cats)
    invest = round(sub["硬件与施工"] + sub["软件"] + sub["实施"], 2)
    om = sub["年运维"]
    payback = invest / total_save
    payback_net = invest / (total_save - om)
    results[name] = dict(cost=cost, sub=sub, invest=invest, om=om,
                         pac_rate=pac_rate, c_rate=c_rate,
                         pac_save=pac_save, c_save=c_save, pam_save=pam_save,
                         sludge_cut_t=sludge_cut_t, sludge_save=sludge_save,
                         power_save=power_save, total_save=total_save,
                         payback=payback, payback_net=payback_net)

# ---------- 打印核对 ----------
for name, r in results.items():
    print("=" * 60)
    print(name, "基线药费合计", round(sum(r["cost"].values()), 1), "万元")
    for cat in cats:
        print(f"  BOM {cat}: {r['sub'][cat]:.2f} 万元")
    print(f"  投资合计 {r['invest']:.2f} 万元 | 年运维 {r['om']:.2f}")
    print(f"  除磷节费 {r['pac_save']:.1f}（节药率 {r['pac_rate']*100:.0f}%）"
          f" 碳源节费 {r['c_save']:.1f}（{r['c_rate']*100:.0f}%）"
          f" PAM {r['pam_save']:.1f} 泥 {r['sludge_save']:.1f}（{r['sludge_cut_t']:.0f}t）"
          f" 电 {r['power_save']:.1f}")
    print(f"  年节省合计 {r['total_save']:.1f} 万元 | 静态回收 {r['payback']:.2f} 年"
          f" | 扣运维 {r['payback_net']:.2f} 年")

# ---------- 出图 ----------
names = list(results.keys())
x = np.arange(len(names))
invest = [results[n]["invest"] for n in names]
saving = [results[n]["total_save"] for n in names]
om = [results[n]["om"] for n in names]
pb = [results[n]["payback"] for n in names]
pbn = [results[n]["payback_net"] for n in names]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.8, 5.4),
                               gridspec_kw={"width_ratios": [1.25, 1]})
w = 0.36
b1 = ax1.bar(x - w/2, invest, w, label="一次性投资", color="#1565c0", zorder=3)
b2 = ax1.bar(x + w/2, saving, w, label="年节省含少产泥与节电", color="#2e7d32", zorder=3)
ax1.bar(x + w/2, om, w, bottom=saving, label="年运维费", color="#ef6c00", alpha=.8, zorder=3)
for rect, v in zip(b1, invest):
    ax1.text(rect.get_x() + rect.get_width()/2, v + 5, f"{v:.0f}", ha="center", fontsize=10)
for rect, v in zip(b2, saving):
    ax1.text(rect.get_x() + rect.get_width()/2, v + 5, f"{v:.0f}", ha="center", fontsize=10)
ax1.set_xticks(x); ax1.set_xticklabels(names, fontsize=10.5)
ax1.set_ylabel("万元")
ax1.set_title("一次性投资 vs 年节省（合成案例实算，以实际报价为准）", fontsize=11.5)
ax1.legend(frameon=False, fontsize=9.5)
ax1.grid(axis="y", alpha=.3)
ax1.set_ylim(0, max(max(invest), max(saving)) * 1.18)

yy = np.arange(len(names))
ax2.barh(yy + 0.18, pb, height=0.34, color="#6a1b9a", label="静态回收期", zorder=3)
ax2.barh(yy - 0.18, pbn, height=0.34, color="#ab47bc", label="扣年运维后回收期", zorder=3)
for i, (a, bv) in enumerate(zip(pb, pbn)):
    ax2.text(a + 0.03, i + 0.18, f"{a:.2f} 年", va="center", fontsize=10)
    ax2.text(bv + 0.03, i - 0.18, f"{bv:.2f} 年", va="center", fontsize=10)
ax2.axvline(2.5, color="#c62828", ls="--", lw=1.2)
ax2.text(2.5, -0.62, "口径上沿 2.5 年", color="#c62828", fontsize=9, ha="center")
ax2.set_yticks(yy); ax2.set_yticklabels(names, fontsize=10.5)
ax2.set_xlabel("年")
ax2.set_xlim(0, 2.9)
ax2.set_title("静态投资回收期（10 万吨级口径 1~2.5 年）", fontsize=11.5)
ax2.legend(frameon=False, fontsize=9.5, loc="upper right")
ax2.grid(axis="x", alpha=.3)

fig.suptitle("三个典型厂方案 BOM 与 ROI 对比（A 轻量盒子 B 标准边缘 C 孪生协同）", fontsize=12.5)
plt.tight_layout()
out = "../images/fig08_04_roi_compare.png"
plt.savefig(out, dpi=130)
print("saved:", out)
