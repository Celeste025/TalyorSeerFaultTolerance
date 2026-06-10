import os
import torch
import matplotlib.pyplot as plt
import numpy as np

def load_steps_from_one_path(path):
    """
    给定一个路径，其内包含 step0.pt, step1.pt, ...
    返回按 step 顺序排列的 tensor list。
    """
    step_files = sorted(
        [f for f in os.listdir(path) if f.startswith("step") and f.endswith(".pt")],
        key=lambda x: int(x.replace("step_", "").replace(".pt", ""))
    )

    tensors = []
    for fname in step_files:
        full_path = os.path.join(path, fname)
        t = torch.load(full_path, map_location="cpu")
        tensors.append(t)

    return tensors


def plot_multi_points_evolution(tensor_list, points, point_names=None,
                                save_path="multi_points.png",
                                figsize=(3.2, 2.8), transparent=True, dpi=700):
    """
    tensor_list: [step0_tensor, step1_tensor, ...]
    points: [(0, 0, 30, 30), (0,5,20,40), ...]
    point_names: 可选，为每个点指定名称
    """

    # ---------- 统一科研论文风格 ----------
    plt.rcParams.update({
        'font.family': 'Liberation Sans',
        'font.size': 12,
        'axes.titlesize': 12,
        'axes.labelsize': 12,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 10,
        'lines.linewidth': 4,
        'lines.markersize': 7,
        'mathtext.fontset': 'stix',
        'axes.labelweight': 'bold',
    })

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    if transparent:
        fig.patch.set_alpha(0.0)
        ax.patch.set_alpha(0.0)

    # ---------- markers & colors ----------
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']

    cmap = plt.cm.Set2
    colors1 = cmap(np.linspace(0, 1, 5))
    colors2 = cmap(np.linspace(0.1, 0.9, 4))
    colors = np.concatenate([colors1, colors2])

    # ---------- 逐点绘制曲线 ----------
    if point_names is None:
        point_names = {pt: f"Point{idx}" for idx, pt in enumerate(points)}

    for i, pt in enumerate(points):
        values = []

        for t in tensor_list:
            if len(pt) > t.dim():
                raise ValueError(f"Point {pt} has more dims than tensor shape {t.shape}")

            index = list(pt) + [slice(None)] * (t.dim() - len(pt))
            v = t[tuple(index)]

            if v.numel() != 1:
                raise ValueError(
                    f"Index {pt} on tensor shape {t.shape} gives non-scalar {v.shape}"
                )

            values.append(v.item())

        steps = list(range(len(values)))
        marker = markers[i % len(markers)]
        color = colors[i]

        label = point_names.get(pt, str(pt))

        ax.plot(steps, values, marker=marker, label=label,
                color=color, markeredgecolor='white',
                markeredgewidth=0.5, alpha=0.9)

    # ---------- 坐标轴 ----------
    ax.set_xlabel("Step")
    ax.set_ylabel("Value")
    ax.xaxis.labelpad = -5
    ax.yaxis.labelpad = -5
    # ---------- legend ----------
    legend = ax.legend(
        loc='upper left',
        frameon=True,
        fancybox=True,
        framealpha=0.3,
        facecolor='white',
        edgecolor='gray',
         # ★ 新增：压紧 legend 内部空隙
        borderpad=0.2,
        labelspacing=0.2,
        handletextpad=0.3,
        ncol=2
    )
    legend.get_frame().set_linewidth(1)
    
    ax.set_ylim(top=2, bottom=-1)
    ax.grid(True, alpha=0.3, linewidth=1)
    ax.tick_params(axis='both', which='major', width=1)

    for spine in ax.spines.values():
        spine.set_linewidth(1)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight',
                    transparent=transparent,
                    facecolor='none' if transparent else 'white',
                    edgecolor='none')
        print(f"图片已保存到: {save_path}")

    plt.close(fig)


if __name__ == "__main__":

    folder = "../PixArt/results_hook/target_PixArtfull_step_30_err_prob_0.0_h_clean/images_gen/layer_out"

    tensor_list = load_steps_from_one_path(folder)

    # 多个点位置
    points = [
        (0, 0, 30, 30),
        (0, 0, 20, 20),
        (0, 0, 40, 10),
        (0, 2, 60, 60),
    ]

    point_names = {
        (0, 0, 30, 30): "P1",
        (0, 0, 20, 20): "P2",
        (0, 0, 40, 10): "P3",
        (0, 2, 60, 60): "P4",
    }

    plot_multi_points_evolution(
        tensor_list,
        points,
        point_names=point_names,
        save_path="multi_points_evolution.svg"
    )
