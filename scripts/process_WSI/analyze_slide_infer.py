import re
import pandas as pd
import matplotlib.pyplot as plt

classes = ['NILM', 'AGC', 'ASC-US', 'LSIL', 'ASC-H', 'HSIL']
cls_map = {
    'NILM': 'NILM',
    'ASC-US': 'ASC-US',
    'LSIL': 'LSIL',
    'ASC-H': 'ASC-H',
    'HSIL': 'HSIL',
    'SCC': 'HSIL',
    'AGC-N': 'AGC',
    'AGC': 'AGC',
    'AGC-NOS': 'AGC',
    'AGC-FN': 'AGC',
}

# ======== 1. 读取并解析数据 ========
txt_file = "log/WS850/hs_round0/infer_result_puretrain.txt"  # 你的txt文件路径
pattern = re.compile(
    r"\[(?P<patientId>.+?)\((?P<slide_clsname>.+?)\)\].*?"
    r"total:(?P<total>\d+),\s*invalid:(?P<invalid>\d+),\s*uncertain:(?P<uncertain>\d+),\s*valid:(?P<valid>\d+),\s*neg:(?P<neg>\d+),\s*pos:(?P<pos>\d+)"
)

rows = []
with open(txt_file, "r", encoding="utf-8") as f:
    for line in f:
        match = pattern.search(line)
        if match:
            data = match.groupdict()
            for k in ["total", "invalid", "uncertain", "valid", "neg", "pos"]:
                data[k] = int(data[k])
            rows.append(data)

df = pd.DataFrame(rows)

# ======== 2. 计算 valid/total ========
df["valid_ratio"] = df["valid"] / df["total"]

# ======== 3. 按 slide_clsname 分析 neg/valid 和 pos/valid ========
df["neg_ratio"] = df["neg"] / df["valid"]
df["pos_ratio"] = df["pos"] / df["valid"]
# 映射 slide_clsname
df["slide_clsname"] = df["slide_clsname"].map(cls_map)
# 按 slide_clsname 分组计算均值
ratio_stats = df.groupby("slide_clsname")[["neg_ratio", "pos_ratio"]].mean()
# 按指定顺序排序
ratio_stats = ratio_stats.reindex(classes)
# 可视化：堆叠柱状图
fig, ax = plt.subplots(figsize=(6, 7))
ax.bar(ratio_stats.index, ratio_stats["neg_ratio"], width=0.5, color="green", label="Neg/Valid")
ax.bar(ratio_stats.index, ratio_stats["pos_ratio"], width=0.5, 
       bottom=ratio_stats["neg_ratio"], color="red", label="Pos/Valid")
# Y 轴留出空白以显示 legend
ax.set_ylim(0, max(ratio_stats["neg_ratio"] + ratio_stats["pos_ratio"]) * 1.2)
# 设置 X 轴
ax.set_xticks(ratio_stats.index)
ax.set_xticklabels(ratio_stats.index, rotation=45)
ax.set_ylabel("Average Ratio")
ax.set_title("Neg/Valid & Pos/Valid by Slide Classname")
ax.legend()
plt.tight_layout()
plt.savefig('log/WS850/hs_round0/infer_puretrain.png')

# ======== 4. 打印 valid/total 最低的 top10 ========
lowest_top10 = df.sort_values("valid_ratio").head(10)
print("Top 10 lowest valid/total ratios:")
print(lowest_top10[["patientId", "slide_clsname", "valid", "total", "valid_ratio"]]
      .to_string(index=False))


'''
Top 10 lowest valid/total ratios:
       patientId slide_clsname  valid  total  valid_ratio
ZY_ONLINE_1_3228          NILM     76   3191     0.023817
ZY_ONLINE_1_3115          NILM     96   3541     0.027111
     JFSW_2_1404         ASC-H     67   2261     0.029633
ZY_ONLINE_1_3118          NILM    130   3541     0.036713
ZY_ONLINE_1_2331          NILM    137   3347     0.040932
ZY_ONLINE_1_2423          NILM    155   3541     0.043773
        WXL_1_87          HSIL    558  11658     0.047864
ZY_ONLINE_1_3507          NILM    204   3423     0.059597
ZY_ONLINE_1_2502          NILM    489   7917     0.061766
ZY_ONLINE_1_2342          NILM    215   3191     0.067377
'''