import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Data preparation
data = [
    {
        'method': 'Q2L', 'input': 224, 'mlc_f1': 28.01, 'fps': 73.92, 
        'color': '#1f77b4', 'marker': 'o',
        'label_cfg': {'offset': (0, -0.5), 'ha': 'center', 'va': 'top'} 
    },
    {
        'method': 'Q2L', 'input': 448, 'mlc_f1': 32.33, 'fps': 18.88, 
        'color': '#1f77b4', 'marker': 's',
        'label_cfg': {'offset': (1, -1.5), 'ha': 'right', 'va': 'center'}
    },
    {
        'method': 'ML-Dec', 'input': 224, 'mlc_f1': 29.77, 'fps': 86.40, 
        'color': '#d69e2e', 'marker': 'o',
        'label_cfg': {'offset': (3, -0.5), 'ha': 'center', 'va': 'top'}
    },
    {
        'method': 'ML-Dec', 'input': 448, 'mlc_f1': 33.17, 'fps': 22.41, 
        'color': '#d69e2e', 'marker': 's',
        'label_cfg': {'offset': (2, 0), 'ha': 'left', 'va': 'center'}
    },
    {
        'method': 'CIPL', 'input': 224, 'mlc_f1': 35.51, 'fps': 78.87, 
        'color': '#5bb75b', 'marker': 'o',
        'label_cfg': {'offset': (3, -0.5), 'ha': 'center', 'va': 'top'}
    },
    {
        'method': 'CIPL', 'input': 448, 'mlc_f1': 36.22, 'fps': 20.19, 
        'color': '#5bb75b', 'marker': 's',
        'label_cfg': {'offset': (2, 0), 'ha': 'left', 'va': 'center'}
    },
    {
        'method': 'Ours', 'input': 1024, 'mlc_f1': 43.90, 'fps': 62.56, 
        'color': '#d62728', 'marker': '*',
        'label_cfg': {'offset': (0, -1), 'ha': 'right', 'va': 'top'} # Ours 的位置
    }
]

# Create figure
plt.figure(figsize=(6, 6))

# 3. 绘制每个数据点
for item in data:
    is_ours = item['method'] == 'Ours'
    size = 500 if is_ours else 150
    zorder = 20 if is_ours else 10
    
    # 绘制散点
    plt.scatter(item['fps'], item['mlc_f1'], 
                color=item['color'], 
                marker=item['marker'], 
                s=size, 
                zorder=zorder, 
                edgecolors='white', 
                linewidth=1.5)

    # 准备标签文本
    if is_ours:
        # Ours 保留两位小数，且加粗
        label_text = f"Ours-1024\n({item['fps']:.2f}, {item['mlc_f1']:.2f})"
        font_weight = 'bold'
    else:
        # 其他保留一位小数即可（或者保持两位）
        label_text = f"{item['method']}-{item['input']}\n({item['fps']:.1f}, {item['mlc_f1']:.1f})"
        font_weight = 'normal'
    
    # --- 关键修改：直接使用 data 中的配置 ---
    cfg = item['label_cfg']
    plt.text(
        item['fps'] + cfg['offset'][0],    # X 坐标 + 偏移
        item['mlc_f1'] + cfg['offset'][1], # Y 坐标 + 偏移
        label_text, 
        fontsize=11, 
        ha=cfg['ha'], 
        va=cfg['va'], 
        fontweight=font_weight, 
        zorder=30
    )

# 4. 装饰图表
plt.title("Performance vs. Efficiency Trade-off", fontsize=18, fontweight='bold', pad=20)
plt.xlabel("Inference Speed (FPS)", fontsize=14)
plt.ylabel("MLC-F1 Score", fontsize=14)
plt.grid(True, linestyle='--', alpha=0.5, zorder=0)

plt.xlim(0, 105)
plt.ylim(25, 48)

# Add arrow for "Better" direction
# style = "Simple, tail_width=0.5, head_width=4, head_length=8"
# kw = dict(arrowstyle=style, color="#c7c7c7", alpha=0.5)
# a3 = patches.FancyArrowPatch((65, 38), (85, 45), connectionstyle="arc3,rad=.2", **kw)
# plt.gca().add_patch(a3)
# plt.text(75, 40, "Better Performance\n& Speed", color="gray", fontsize=12, ha='center')

# Custom Legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#1f77b4', label='Q2L-224', markersize=10),
    Line2D([0], [0], marker='s', color='w', markerfacecolor='#1f77b4', label='Q2L-448', markersize=10),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#d69e2e', label='ML-Dec-224', markersize=10),
    Line2D([0], [0], marker='s', color='w', markerfacecolor='#d69e2e', label='ML-Dec-448', markersize=10),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#5bb75b', label='CIPL-224', markersize=10),
    Line2D([0], [0], marker='s', color='w', markerfacecolor='#5bb75b', label='CIPL-448', markersize=10),
    Line2D([0], [0], marker='*', color='w', markerfacecolor='#d62728', label='Ours-1024', markersize=15)
]
plt.legend(handles=legend_elements, loc='upper right', framealpha=0.95, title='Model-Input Size')

plt.tight_layout()
plt.savefig('statistic_results/performance_tradeoff_plot.png', dpi=300)