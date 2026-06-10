import matplotlib.pyplot as plt
import numpy as np
import os

def plot_mem_compute_bar(models, mem, mem_p, compute, compute_p,
                         save_path=None, figsize=(3.8,2.8), dpi=700):
    """
    绘制科研风格柱图：每个模型两根柱子，下半部分为 mem，上半部分为 compute。

    参数
    -----
    models : list[str]
        模型名称，对应 x 轴每个位置。
    mem : array-like
        第一根（左）柱子对应的内存相关功耗/能耗（与 compute 同量纲）。
    compute : array-like
        第一根（左）柱子对应的计算相关功耗/能耗。
    mem_p : array-like
        第二根（右）柱子对应的内存相关功耗/能耗（如带保护/另一种配置）。
    compute_p : array-like
        第二根（右）柱子对应的计算相关功耗/能耗。

    每根柱子会按 (mem, compute) 或 (mem_p, compute_p) 归一化到总高度 1，
    只比较各模型中 mem 与 compute 的占比；左右两根柱子用于对比两种配置
    （如 baseline vs 带容错/保护）。
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import os

    # ----------------- 科研风格参数 -----------------
    plt.rcParams.update({
        'font.family': 'Liberation Sans',
        'font.size': 12,
        'axes.titlesize': 12,
        'axes.labelsize': 12,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 10,
        'lines.linewidth': 2,
        'lines.markersize': 6,
        'mathtext.fontset': 'stix',
        'axes.labelweight': 'bold',
    })
    cmap = plt.cm.Set2
    colors = cmap.colors
    color_compute, color_mem = colors[0], colors[5] 

    # ----------------- 计算归一化比例 -----------------
    f = mem + compute
    f_p = mem_p + compute_p

    # 下半部分mem / 上半部分compute
    mem_ratio_1     = mem / f
    compute_ratio_1 = compute / f
    mem_ratio_2     = mem_p / f
    compute_ratio_2 = compute_p / f

    # x轴位置
    x = np.arange(len(models))
    width = 0.3  # 柱子稍微细一点
    spacing = 0.6
    # ----------------- 绘图 -----------------
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    # 第一根柱子
    ax.bar(x - width * spacing, mem_ratio_1, width, color=color_mem)
    ax.bar(x - width * spacing, compute_ratio_1, width, bottom=mem_ratio_1, color=color_compute)
    # 第二根柱子，留空
    ax.bar(x + width * spacing, mem_ratio_2, width, color=color_mem)
    ax.bar(x + width * spacing, compute_ratio_2, width, bottom=mem_ratio_2, color=color_compute)

    # 单位高度100：黄段顶部标该段高度，绿段内标黄+绿总高度；标注用更深绿色
    unit = 100
    fs = 8  # 标注字体大小
    label_color = '#0a3d0a'  # 更深绿色
    for i in range(len(models)):
        # 第一根柱子：下半段(黄)顶部标该段高度；上半段(绿)内标总高度100，文字在上边线下方
        x_left = x[i] - width * spacing
        ax.text(x_left, mem_ratio_1[i], f'{mem_ratio_1[i] * unit:.1f}', ha='center', va='bottom', fontsize=fs, color=label_color)
        ax.text(x_left, 1.0, f'{unit:.1f}', ha='center', va='top', fontsize=fs, color=label_color)  # 总高度，柱内
        # 第二根柱子
        x_right = x[i] + width * spacing
        ax.text(x_right, mem_ratio_2[i], f'{mem_ratio_2[i] * unit:.1f}', ha='center', va='bottom', fontsize=fs, color=label_color)
        ax.text(x_right, mem_ratio_2[i] + compute_ratio_2[i], f'{(mem_ratio_2[i] + compute_ratio_2[i]) * unit:.1f}', ha='center', va='top', fontsize=fs, color=label_color)  # 总高度，柱内

    # x轴和y轴标签
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontweight='bold')
    ax.set_ylabel('Normalized Power', fontweight='bold')

    # y轴最大值
    ax.set_ylim(0, 1.4)

    # ----------------- 图例 -----------------
    ax.legend(['Mem', 'Compute', ], loc='upper left',
              frameon=True, fancybox=True, framealpha=0.4,
              facecolor='white', edgecolor='gray')
    for spine in ax.spines.values():
        spine.set_linewidth(1)  # 边框线宽

    # 网格
    ax.grid(True, alpha=0.3, linewidth=1)
    ax.tick_params(axis='both', which='major', width=1)
    ax.minorticks_off()

    plt.tight_layout()

    # 保存
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight',
                    transparent=True, facecolor='none', edgecolor='none')
        print(f"图片已保存到: {save_path}")

    plt.close(fig)


if __name__ == "__main__":
    models = ['DiT-XL512', 'PixArt-α', 'SD1.5']
    mem      = np.array([1.00, 1.00, 1.00])
    mem_p    = np.array([1.2, 1.8, 1.18])
    compute  = np.array([5.81, 39.9, 6.61])
    compute_p= np.array([3.17, 23.48, 4.01])

    plot_mem_compute_bar(models, mem, mem_p, compute, compute_p,
                        save_path='power_break_down.svg')
    plot_mem_compute_bar(models, mem, mem_p, compute, compute_p,
                        save_path='power_break_down.png')

    print("Img saved as power_break_down.png")