import matplotlib.pyplot as plt
import numpy as np
import os

def plot_two_bars(
    values,
    labels=("Row-major", "Repack"),
    ylabel="DRAM row activations",
    save_path=None,
    figsize=(2.6, 2.8),
    transparent=True,
    dpi=700,
    bar_width=0.55,  # 柱子宽度
    xtick_labelpad=2  # x轴刻度标签与柱子间距
):
    # 全局科研风格
    plt.rcParams.update({
        'font.family': 'Liberation Sans',
        'font.size': 12,
        'axes.titlesize': 12,
        'axes.labelsize': 12,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 10,
        'lines.linewidth': 4,
        'mathtext.fontset': 'stix',
        'axes.labelweight': 'bold',
    })

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    if transparent:
        fig.patch.set_alpha(0.0)
        ax.patch.set_alpha(0.0)

    # 两个 bar
    x = np.arange(len(values))
    cmap = plt.cm.Set2
    colors = [cmap.colors[0], cmap.colors[2]]

    bars = ax.bar(
        x, values,
        color=colors,
        edgecolor='white',
        linewidth=0.8,
        width=bar_width
    )

    # x 轴标签显示在柱子下方，稍微紧凑
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0, ha='center', rotation_mode='anchor', fontweight='bold')
    ax.tick_params(axis='x', pad=6) 


    # y 轴标签
    ax.set_ylabel(ylabel, labelpad=4)

    # 设置 y 最大值 = 最大 value * 1.3
    ax.set_ylim(0, max(values) * 1.1)

    # 边框
    for spine in ax.spines.values():
        spine.set_linewidth(1)

    # 网格
    ax.grid(axis='y', alpha=0.3, linewidth=1)

    # 无图例
    # plt.legend()  # 已取消

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(
            save_path,
            dpi=dpi,
            bbox_inches='tight',
            transparent=transparent,
            facecolor='none' if transparent else 'white',
            edgecolor='none'
        )
        print(f"图片已保存到: {save_path}")

    plt.close(fig)


if __name__ == "__main__":
    plot_two_bars(
        values=[3*32*32, 32*32/8],
        save_path="ablation_dram_acess.svg"
    )
