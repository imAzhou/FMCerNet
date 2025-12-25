import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
import matplotlib.gridspec as gridspec  # 引入 GridSpec 模块
import numpy as np
import os

# ==========================================
# 1. 数据准备 (保持不变)
# ==========================================
all_data = [
    # --- Row 1: CDetector ---
    {
        'title': 'CDetector - Train',
        'nilm': 28776,
        'positive': 36690,
        'pos_data': {'ASCUS': 3666, 'LSIL': 3481, 'ASCH': 5044, 'HSIL': 21979, 'AGC': 6454},
        'row': 0, 'col': 0 
    },
    {
        'title': 'CDetector - Test',
        'nilm': 3571,
        'positive': 4117,
        'pos_data': {'ASCUS': 413, 'LSIL': 393, 'ASCH': 557, 'HSIL': 2289, 'AGC': 869},
        'row': 0, 'col': 1 
    },
    # --- Row 2: L-CerScan ---
    {
        'title': 'L-CerScan - Train',
        'nilm': 79419,
        'positive': 66158,
        'pos_data': {'ASCUS': 25533, 'LSIL': 7365, 'ASCH': 23412, 'HSIL': 8076, 'AGC': 11863},
        'row': 1, 'col': 0 
    },
    {
        'title': 'L-CerScan - Test',
        'nilm': 20278,
        'positive': 16202,
        'pos_data': {'ASCUS': 6509, 'LSIL': 1917, 'ASCH': 4628, 'HSIL': 1989, 'AGC': 3711},
        'row': 1, 'col': 1 
    }
]

# ==========================================
# 2. 辅助函数
# ==========================================
def make_autopct(values):
    def my_autopct(pct):
        total = sum(values)
        val = int(round(pct * total / 100.0))
        return '{p:.1f}%\n({v:,})'.format(p=pct, v=val)
    return my_autopct

# ==========================================
# 3. 绘图核心函数
# ==========================================
def plot_pie_bar_unit(fig, inner_gs, data_info):
    title = data_info['title']
    nilm_count = data_info['nilm']
    pos_total = data_info['positive']
    pos_data = data_info['pos_data']
    
    pos_values = list(pos_data.values())
    pos_labels = list(pos_data.keys())
    pie_counts = [nilm_count, pos_total]
    pie_labels = ['Negative', 'Positive']
    
    # === 关键修改：从传入的 inner_gs 中获取子图 ===
    # inner_gs[0] 是饼图位置，inner_gs[1] 是柱状图位置
    ax_pie = fig.add_subplot(inner_gs[0])
    ax_bar = fig.add_subplot(inner_gs[1])

    # --- Pie Plot ---
    pie_colors = ['#AEC7E8', '#FF9896'] 
    explode = (0, 0.05)
    
    wedges, texts, autotexts = ax_pie.pie(
        pie_counts, 
        autopct=make_autopct(pie_counts),
        startangle=90, 
        colors=pie_colors, 
        explode=explode, 
        radius=0.9, # 饼图稍微大一点点，贴近边界
        pctdistance=0.5, 
        textprops={'fontsize': 8.5, 'weight': 'bold'}
    )
    
    # ax_pie.set_title(title, fontsize=13, fontweight='bold', pad=20)
    # ax_pie.legend(wedges, pie_labels, loc="lower left", 
    #               bbox_to_anchor=(-0.1, -0.1), fontsize=8, frameon=False)

    # --- Bar Plot ---
    bar_width = 0.4
    step = 0.6 
    x_pos = np.arange(len(pos_labels)) * step
    
    bar_colors = plt.cm.Reds(np.linspace(0.4, 0.9, len(pos_labels)))
    
    bars = ax_bar.bar(x_pos, pos_values, width=bar_width, 
                      color=bar_colors, edgecolor='black', alpha=0.9)
    
    ax_bar.set_xticks(x_pos)
    ax_bar.set_xticklabels(pos_labels, fontsize=8.5)
    ax_bar.tick_params(axis='y', labelsize=8.5)
    ax_bar.grid(axis='y', linestyle='--', alpha=0.5)
    # ax_bar.set_title("Positive Detail", fontsize=10, style='italic')

    ax_bar.set_xlim(min(x_pos) - 0.4, max(x_pos) + 0.4)
    ax_bar.set_ylim(0, max(pos_values) * 1.15)

    for bar in bars:
        height = bar.get_height()
        ax_bar.annotate(f'{height:,}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 2), textcoords="offset points",
                        ha='center', va='bottom', fontsize=8.5)

    # --- Connections ---
    theta1, theta2 = wedges[1].theta1, wedges[1].theta2
    center, r = wedges[1].center, wedges[1].r

    # 连接线逻辑保持不变，因为是跨子图坐标系
    x_p_top = r * np.cos(np.radians(theta2)) + center[0]
    y_p_top = r * np.sin(np.radians(theta2)) + center[1]
    con1 = ConnectionPatch(xyA=(-0.4, ax_bar.get_ylim()[1]), coordsA=ax_bar.transData,
                           xyB=(x_p_top, y_p_top), coordsB=ax_pie.transData,
                           color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    
    x_p_bot = r * np.cos(np.radians(theta1)) + center[0]
    y_p_bot = r * np.sin(np.radians(theta1)) + center[1]
    con2 = ConnectionPatch(xyA=(-0.4, 0), coordsA=ax_bar.transData,
                           xyB=(x_p_bot, y_p_bot), coordsB=ax_pie.transData,
                           color="black", linestyle="--", linewidth=0.8, alpha=0.5)

    fig.add_artist(con1)
    fig.add_artist(con2)

# ==========================================
# 4. 主程序布局 (彻底重构)
# ==========================================
fig = plt.figure(figsize=(16, 10)) 

# 1. 创建外层网格：2行2列 (控制四大块的布局)
#    wspace=0.25 保证左右两组(Train/Test)之间有足够的缝隙
outer_gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1], wspace=0.25, hspace=0.3)

for info in all_data:
    row = info['row']
    col = info['col']
    
    # 2. 获取当前大格子的位置 spec
    cell_spec = outer_gs[row, col]
    
    # 3. 在这个大格子内部，创建一个 1x2 的内层网格 (SubplotSpec)
    #    关键点：wspace=0.0 让饼图和柱状图贴在一起！
    inner_gs = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=cell_spec, 
                                                width_ratios=[1, 1.2], # 饼图:柱状图 = 1:1.2
                                                wspace=0.1)            # <--- 这里是控制距离的关键！
    
    # 传递这个 inner_gs 给绘图函数
    plot_pie_bar_unit(fig, inner_gs, info)

output_path = 'dataset_dist_v7_nested_grid.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"图片已生成: {os.path.abspath(output_path)}")
# plt.show()