import os
import torch
import matplotlib.pyplot as plt
import numpy as np

def load_steps_from_paths(paths):
    """
    给定多个路径，每个路径内包含 step0.pt, step1.pt, ...
    返回:
        data = {
            path1: [tensor_step0, tensor_step1, ...],
            path2: [...],
        }
    """
    all_data = {}

    for p in paths:
        step_files = sorted(
            [f for f in os.listdir(p) if f.startswith("step") and f.endswith(".pt")],
            key=lambda x: int(x.replace("step_", "").replace(".pt", ""))
        )

        tensors = []
        for fname in step_files:
            full_path = os.path.join(p, fname)
            t = torch.load(full_path, map_location="cpu")
            tensors.append(t)

        all_data[p] = tensors
    
    return all_data

def plot_point_evolution(all_data, point, path_names=None, save_path="observation4_Pix.png", 
                        figsize=(3.8, 2.8), transparent=True, dpi=300):
    """
    point: 单个point坐标，例如：(0, 0, 30, 30)
    all_data：来自 load_steps_from_paths 的字典
    path_names: 可选的字典，为每个path指定显示名称，例如：
        {path1: "Clean", path2: "Error 0.5", ...}
        如果为None，则使用路径的basename
    """

    # 设置科研论文风格的参数
    plt.rcParams.update({
        'font.family': 'Liberation Sans',     
        'font.size': 12,           # 基础字体大小
        'axes.titlesize': 12,      # 标题字体大小
        'axes.labelsize': 12,      # 坐标轴标签字体大小
        'xtick.labelsize': 12,     # x轴刻度标签字体大小
        'ytick.labelsize': 12,     # y轴刻度标签字体大小
        'legend.fontsize': 10,     # 图例字体大小
        'lines.linewidth': 4,      # 线宽
        'lines.markersize': 7,     # 标记大小
        'mathtext.fontset': 'stix', # 数学字体
        'axes.labelweight': 'bold', # 坐标轴标签加粗
    })
    
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # 设置透明背景
    if transparent:
        fig.patch.set_alpha(0.0)
        ax.patch.set_alpha(0.0)
    
    # 定义标记和颜色
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    # colors = plt.cm.Set2(np.linspace(0, 1, len(all_data)))
    cmap = plt.cm.Set2
    colors1 = cmap(np.linspace(0, 1, 5))      # 第 1 组 5 个
    colors2 = cmap(np.linspace(0.1, 0.9, 4))  # 第 2 组 4 个（无重复）
    colors = np.concatenate([colors1, colors2])
    
    # 处理路径名称
    if path_names is None:
        path_names = {}
        for path in all_data.keys():
            path_names[path] = os.path.basename(path)
    
    # 绘图：每个path一条线
    for i, (path, tensor_list) in enumerate(all_data.items()):
        values = []

        for t in tensor_list:
            # 构造用于索引的 key：point 定义的维度 + 对剩余维度使用 slice(None)
            if len(point) > t.dim():
                raise ValueError(f"Point {point} has more dims than tensor with shape {t.shape}")

            # 比如 tensor.shape = (C,H,W), point = (2,10) -> (2,10, :)
            index = list(point) + [slice(None)] * (t.dim() - len(point))

            v = t[tuple(index)]

            # 如果切片后 v 还是一个 tensor（比如取了一个行/一个通道），那必须 reduce 成标量
            if v.numel() != 1:
                raise ValueError(
                    f"Index {point} on tensor shape {t.shape} results in non-scalar "
                    f"tensor with shape {v.shape}. Please provide full coordinate."
                )

            values.append(v.item())

        steps = list(range(len(values)))
        marker = markers[i % len(markers)]
        color = colors[i]
        
        # 使用手动指定的名称或默认名称
        label = path_names.get(path, os.path.basename(path))
        
        # 使用统一的绘图风格
        # steps = steps[10:]
        # values = values[10:]
        ax.plot(steps, values, marker=marker, label=label, color=color,
               markeredgecolor='white', markeredgewidth=0.5, alpha=0.9)

    # 设置标签（移除大标题）
    ax.set_xlabel("Step")
    ax.set_ylabel("Value")
    
    # 设置图例
    legend = ax.legend(
        loc='upper left',
        frameon=True,
        fancybox=True,
        framealpha=0.3,
        facecolor='white',
        edgecolor='gray'
    )
    legend.get_frame().set_linewidth(1)
    
    # 网格和边框设置
    ax.grid(True, alpha=0.3, linewidth=1)
    ax.tick_params(axis='both', which='major', width=1)
    
    # 设置边框线宽
    for spine in ax.spines.values():
        spine.set_linewidth(1)
    
    plt.tight_layout()
    
    # 保存图片
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight', 
                   transparent=transparent,
                   facecolor='none' if transparent else 'white',
                   edgecolor='none')
        print(f"图片已保存到: {save_path}")
        print("\n")
    
    plt.close(fig)


if __name__ == "__main__":
    paths = [
        "../PixArt/results_hook/target_PixArtfull_step_30_err_prob_0.0_h_clean/images_gen/layer_out",
        "../PixArt/results_hook/target_PixArtfull_step_30_err_prob_0.0_error_0.5/images_gen/layer_out",
        "../PixArt/results_hook/target_PixArtfull_step_30_err_prob_0.0_error_1/images_gen/layer_out",
        "../PixArt/results_hook/target_PixArtfull_step_30_err_prob_0.0_error_2/images_gen/layer_out",
        "../PixArt/results_hook/target_PixArtfull_step_30_err_prob_0.0_error_4/images_gen/layer_out",
    ]

    all_data = load_steps_from_paths(paths)

    # 手动指定每个path的显示名称
    path_names = {
        "../PixArt/results_hook/target_PixArtfull_step_30_err_prob_0.0_h_clean/images_gen/layer_out": "Clean",
        "../PixArt/results_hook/target_PixArtfull_step_30_err_prob_0.0_error_0.5/images_gen/layer_out": "Error=0.5",
        "../PixArt/results_hook/target_PixArtfull_step_30_err_prob_0.0_error_1/images_gen/layer_out": "Error=1",
        "../PixArt/results_hook/target_PixArtfull_step_30_err_prob_0.0_error_2/images_gen/layer_out": "Error=2",
        "../PixArt/results_hook/target_PixArtfull_step_30_err_prob_0.0_error_4/images_gen/layer_out": "Error=4",
    }
    plot_point_evolution(all_data, (0, 0, 30, 30), path_names=path_names, save_path="observation4_PixArt.svg")


    paths = [
        "../TaylorSeer-DiT/results_hook/target_DiTXL512full_step_50_err_prob_0.0_clean/images_gen/layer_out",
        "../TaylorSeer-DiT/results_hook/target_DiTXL512full_step_50_err_prob_0.0_error_0.5/images_gen/layer_out",
        "../TaylorSeer-DiT/results_hook/target_DiTXL512full_step_50_err_prob_0.0_error_1/images_gen/layer_out",
        "../TaylorSeer-DiT/results_hook/target_DiTXL512full_step_50_err_prob_0.0_error_2/images_gen/layer_out",
        "../TaylorSeer-DiT/results_hook/target_DiTXL512full_step_50_err_prob_0.0_error_4/images_gen/layer_out",
    ]
    all_data = load_steps_from_paths(paths)
    path_names = {
        "../TaylorSeer-DiT/results_hook/target_DiTXL512full_step_50_err_prob_0.0_clean/images_gen/layer_out": "Clean",
        "../TaylorSeer-DiT/results_hook/target_DiTXL512full_step_50_err_prob_0.0_error_0.5/images_gen/layer_out": "Error=0.5",
        "../TaylorSeer-DiT/results_hook/target_DiTXL512full_step_50_err_prob_0.0_error_1/images_gen/layer_out": "Error=1",
        "../TaylorSeer-DiT/results_hook/target_DiTXL512full_step_50_err_prob_0.0_error_2/images_gen/layer_out": "Error=2",
        "../TaylorSeer-DiT/results_hook/target_DiTXL512full_step_50_err_prob_0.0_error_4/images_gen/layer_out": "Error=4"
    }
    plot_point_evolution(all_data, (0, 0, 16, 16), path_names=path_names, save_path="observation4_DiT.svg")
    