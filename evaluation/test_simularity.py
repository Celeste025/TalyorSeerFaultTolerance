# import os
# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
# import gc
# from tqdm import tqdm

# # ====== 参数配置 ======
# records_dir = "/data/home/jinqiwen/workspace/diffusion_fault_tolerance/TaylorSeerFaultTolerance/TaylorSeers-Diffusers/taylorseer_flux/results/Debug2_target_Skip_step_50_err_prob_0.0_h/images_gen/layer_inout"
# save_dir = os.path.join(os.path.dirname(records_dir), "analysis_results")
# print(save_dir)
# os.makedirs(save_dir, exist_ok=True)

# # ====== 获取 step 文件 ======
# files = sorted([f for f in os.listdir(records_dir) if f.endswith(".pt")])
# print(f"找到 {len(files)} 个步骤文件")

# if len(files) < 2:
#     raise ValueError("至少需要2个step才能计算变化趋势。")

# # ====== 初始化 ======
# prev_data = torch.load(os.path.join(records_dir, files[0]), map_location="cpu")
# layer_names = list(prev_data.keys())
# l1_diffs = {layer: [] for layer in layer_names}

# # ====== 流式分析相邻 step 的 L1 输出变化 ======
# for i in tqdm(range(len(files) - 1), desc="Processing steps"):
#     path_next = os.path.join(records_dir, files[i + 1])
#     next_data = torch.load(path_next, map_location="cpu")

#     for layer in layer_names:
#         if layer not in prev_data or layer not in next_data:
#             l1_diffs[layer].append(np.nan)
#             continue

#         out_t = prev_data[layer].get("output", None)
#         out_t1 = next_data[layer].get("output", None)

#         if out_t is None or out_t1 is None:
#             l1_diffs[layer].append(np.nan)
#             continue

#         # 如果 output 是 tuple，取第一个 tensor
#         if isinstance(out_t, tuple): out_t = out_t[0]
#         if isinstance(out_t1, tuple): out_t1 = out_t1[0]

#         if not (isinstance(out_t, torch.Tensor) and isinstance(out_t1, torch.Tensor)):
#             l1_diffs[layer].append(np.nan)
#             continue
#         if out_t.shape != out_t1.shape:
#             l1_diffs[layer].append(np.nan)
#             continue

#         diff = torch.mean(torch.abs(out_t - out_t1)).item()
#         l1_diffs[layer].append(diff)

#     if (i + 1) % 5 == 0 or i == len(files) - 2:
#         example_layer = layer_names[0]
#         print(f"[Step {i+1}/{len(files)-1}] 示例层 '{example_layer}' 当前 L1: {l1_diffs[example_layer][-1]:.6f}")

#     # 释放内存
#     del prev_data
#     gc.collect()
#     torch.cuda.empty_cache()

#     prev_data = next_data

# # ====== ✅ 保存结果为 CSV ======
# df = pd.DataFrame(l1_diffs)
# df.index = [f"step_{i}" for i in range(len(df))]
# csv_path = os.path.join(save_dir, "layer_l1_diffs.csv")
# df.to_csv(csv_path)
# print(f"✅ 已保存层输出变化结果到: {csv_path}")

# # ===================================================
# # ============== 绘图函数封装 =======================
# # ===================================================

# def plot_layer_trends(df, save_dir, group_size=10):
#     num_layers = len(df.columns)
#     num_groups = (num_layers + group_size - 1) // group_size

#     for g in range(num_groups):
#         start = g * group_size
#         end = min((g + 1) * group_size, num_layers)
#         sub_layers = df.columns[start:end]

#         plt.figure(figsize=(10, 5))
#         for layer in sub_layers:
#             plt.plot(df.index, df[layer], label=layer, alpha=0.7, linewidth=0.8)
#         plt.xlabel("Step")
#         plt.ylabel("Mean L1 Change (output)")
#         plt.title(f"Layer Output Change vs Step (Layers {start}–{end-1})")
#         plt.legend(fontsize=7, bbox_to_anchor=(1.05, 1), loc="upper left")
#         plt.tight_layout()
#         out_path = os.path.join(save_dir, f"layer_l1_trends_part{g+1}.png")
#         plt.savefig(out_path, dpi=300)
#         plt.close()
#         print(f"📈 子图已保存: {out_path}")

# def plot_layer_correlation(df, save_dir):
#     corr = df.corr(method="pearson")
#     plt.figure(figsize=(10, 8))
#     plt.imshow(corr, cmap="coolwarm", interpolation="nearest")
#     plt.colorbar(label="Pearson Corr")
#     plt.title("Layer Dynamics Correlation")
#     plt.xticks(range(len(corr.columns)), corr.columns, rotation=90, fontsize=6)
#     plt.yticks(range(len(corr.index)), corr.index, fontsize=6)
#     plt.tight_layout()
#     out_path = os.path.join(save_dir, "layer_correlation_heatmap.png")
#     plt.savefig(out_path, dpi=300)
#     plt.close()
#     print(f"🔥 层间相关性热力图已保存: {out_path}")

# # ===================================================
# # ============== 调用绘图函数 =======================
# # ===================================================
# plot_layer_trends(df, save_dir, group_size=10)
# plot_layer_correlation(df, save_dir)



import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import gc
from tqdm import tqdm

# ====== 参数配置 ======
records_dir = "/data/home/jinqiwen/workspace/diffusion_fault_tolerance/TaylorSeerFaultTolerance/TaylorSeers-Diffusers/taylorseer_flux/results/Debug2_target_Skip_step_50_err_prob_0.0_h/images_gen/layer_inout"
save_dir = os.path.join(os.path.dirname(records_dir), "analysis_results")
print(save_dir)
os.makedirs(save_dir, exist_ok=True)

# ====== 指定要分析的层名称 ======
target_layer = "transformer_blocks.13"  # 请替换为实际的层名称

# ====== 获取 step 文件 ======
files = sorted([f for f in os.listdir(records_dir) if f.endswith(".pt")])
print(f"找到 {len(files)} 个步骤文件")

if len(files) == 0:
    raise ValueError("没有找到任何步骤文件。")

def get_layer_tensor(layer_name, step_files, records_dir, max_steps=None):
    """
    流式读取指定层的输出tensor
    
    Args:
        layer_name: 目标层名称
        step_files: 步骤文件列表
        records_dir: 记录文件目录
        max_steps: 最大步骤数（用于限制内存使用）
    
    Returns:
        tensors: 各步骤的tensor列表
        valid_steps: 有效的步骤索引列表
    """
    tensors = []
    valid_steps = []
    
    if max_steps is None:
        max_steps = len(step_files)
    
    for i in tqdm(range(min(len(step_files), max_steps)), desc=f"读取层 '{layer_name}' 数据"):
        file_path = os.path.join(records_dir, step_files[i])
        try:
            data = torch.load(file_path, map_location="cpu")
            if layer_name not in data:
                print(f"警告: 步骤 {i} 中未找到层 '{layer_name}'")
                continue
                
            layer_data = data[layer_name]
            output_tensor = layer_data.get("output", None)
            
            if output_tensor is None:
                print(f"警告: 步骤 {i} 中层 '{layer_name}' 没有输出")
                continue
                
            # 如果 output 是 tuple，取第一个 tensor
            if isinstance(output_tensor, tuple):
                output_tensor = output_tensor[0]
            
            if not isinstance(output_tensor, torch.Tensor):
                print(f"警告: 步骤 {i} 的输出不是tensor")
                continue
            
            # 如果维度大于2，去掉第一维（通常是batch维度）
            if output_tensor.dim() > 2:
                output_tensor = output_tensor[0]  # 取batch中的第一个样本
            
            tensors.append(output_tensor.detach().cpu())
            valid_steps.append(i)
            
            # 释放内存
            del data
            gc.collect()
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"错误: 读取步骤 {i} 时出错: {e}")
            continue
    
    return tensors, valid_steps

def plot_tensor_heatmaps(tensors, valid_steps, save_path, max_plots=20, vmin=None, vmax=None, 
                        region=None):
    """
    绘制每个步骤tensor的热图，可手动设置上下限并统计超出范围的数量，支持区域选择
    
    Args:
        tensors: 各步骤的tensor列表
        valid_steps: 有效的步骤索引列表
        save_path: 保存路径
        max_plots: 最大绘图数量
        vmin: 颜色映射下限
        vmax: 颜色映射上限
        region: 区域选择，格式为 (x_start, y_start, dx, dy)
                如果为None，则绘制整个tensor
    """
    if not tensors:
        print("没有可用的tensor数据")
        return
    
    # 检查第一个tensor的形状
    base_tensor = tensors[0]
    print(f"原始tensor形状: {base_tensor.shape}")
    
    # 处理区域选择
    if region is not None:
        x_start, y_start, dx, dy = region
        x_end = x_start + dx
        y_end = y_start + dy
        
        # 验证区域是否有效
        if x_start < 0 or y_start < 0 or x_end > base_tensor.shape[0] or y_end > base_tensor.shape[1]:
            print(f"警告: 区域 {region} 超出tensor范围 {base_tensor.shape}")
            # 自动调整到有效范围
            x_start = max(0, x_start)
            y_start = max(0, y_start)
            x_end = min(base_tensor.shape[0], x_end)
            y_end = min(base_tensor.shape[1], y_end)
            dx = x_end - x_start
            dy = y_end - y_start
            print(f"已调整区域为: ({x_start}, {y_start}, {dx}, {dy})")
        
        print(f"分析区域: [{x_start}:{x_end}, {y_start}:{y_end}]")
    else:
        print("分析整个tensor")
    
    # 提取区域数据用于范围计算
    if region is not None:
        region_tensors = []
        for tensor in tensors:
            if tensor.dim() == 2:
                region_tensor = tensor[x_start:x_end, y_start:y_end]
            else:
                # 对于非2D tensor，先reshape再切片
                tensor_2d = tensor.view(tensor.size(0), -1)
                region_tensor = tensor_2d[x_start:x_end, y_start:y_end]
            region_tensors.append(region_tensor)
        
        # 使用区域数据计算范围
        all_values = torch.cat([t.flatten() for t in region_tensors])
    else:
        all_values = torch.cat([t.flatten() for t in tensors])
    
    # 如果没有手动设置上下限，自动计算合理的范围
    if vmin is None or vmax is None:
        auto_vmin = all_values.quantile(0.01).item()  # 1%分位数
        auto_vmax = all_values.quantile(0.99).item()  # 99%分位数
        
        if vmin is None:
            vmin = auto_vmin
        if vmax is None:
            vmax = auto_vmax
        
        print(f"自动计算的颜色范围: [{vmin:.2f}, {vmax:.2f}]")
    
    # 限制绘图数量以避免内存问题
    plot_indices = np.linspace(0, len(tensors)-1, min(len(tensors), max_plots), dtype=int)
    
    n_rows = int(np.ceil(len(plot_indices) / 5))
    n_cols = min(5, len(plot_indices))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])
    elif n_rows > 1 and n_cols > 1:
        axes = axes.flatten()
    
    # 统计总体信息
    total_out_of_range_stats = {
        'below': 0, 'above': 0, 'total_elements': 0
    }
    
    for idx, (ax, tensor_idx) in enumerate(zip(axes, plot_indices)):
        tensor = tensors[tensor_idx]
        step = valid_steps[tensor_idx]
        
        # 处理tensor形状和区域选择
        if tensor.dim() == 1:
            size = tensor.size(0)
            factors = [i for i in range(1, int(np.sqrt(size)) + 1) if size % i == 0]
            if factors:
                h = factors[-1]
                w = size // h
                tensor_2d = tensor.reshape(h, w)
            else:
                h = int(np.sqrt(size))
                w = int(np.ceil(size / h))
                tensor_2d = tensor[:h*w].reshape(h, w)
        elif tensor.dim() == 2:
            tensor_2d = tensor
        else:
            tensor_2d = tensor.view(tensor.size(0), -1)
        
        # 应用区域选择
        if region is not None:
            tensor_2d = tensor_2d[x_start:x_end, y_start:y_end]
        
        tensor_np = tensor_2d.numpy()
        flat_tensor = tensor_np.flatten()
        
        # 统计超出范围的数量
        below_count = np.sum(flat_tensor < vmin)
        above_count = np.sum(flat_tensor > vmax)
        total_elements = flat_tensor.size
        
        # 更新总体统计
        total_out_of_range_stats['below'] += below_count
        total_out_of_range_stats['above'] += above_count
        total_out_of_range_stats['total_elements'] += total_elements
        
        below_percent = (below_count / total_elements) * 100
        above_percent = (above_count / total_elements) * 100
        
        # 绘制热图
        im = ax.imshow(tensor_np, cmap='viridis', aspect='auto', vmin=vmin, vmax=vmax)
        
        # 添加颜色条
        plt.colorbar(im, ax=ax, shrink=0.8)
        
        # 设置标题和统计信息
        if region is not None:
            title = f'Step {step}\nRegion: [{x_start}:{x_end}, {y_start}:{y_end}]'
        else:
            title = f'Step {step}\nShape: {tensor_2d.shape}'
        ax.set_title(title, fontsize=10)
        
        # 在子图旁边添加统计信息
        stats_text = (f'Range: [{vmin:.1f}, {vmax:.1f}]\n'
                     f'Below: {below_count} ({below_percent:.1f}%)\n'
                     f'Above: {above_count} ({above_percent:.1f}%)\n'
                     f'Min: {flat_tensor.min():.1f}\n'
                     f'Max: {flat_tensor.max():.1f}')
        
        # 在子图右侧添加文本
        ax.text(1.05, 0.5, stats_text, transform=ax.transAxes, fontsize=8,
               verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # 隐藏多余的子图
    for idx in range(len(plot_indices), len(axes)):
        axes[idx].set_visible(False)
    
    # 添加总体统计信息
    total_below_percent = (total_out_of_range_stats['below'] / total_out_of_range_stats['total_elements']) * 100
    total_above_percent = (total_out_of_range_stats['above'] / total_out_of_range_stats['total_elements']) * 100
    
    overall_stats = (f'Overall Statistics:\n'
                    f'Total elements: {total_out_of_range_stats["total_elements"]:,}\n'
                    f'Below {vmin}: {total_out_of_range_stats["below"]:,} ({total_below_percent:.2f}%)\n'
                    f'Above {vmax}: {total_out_of_range_stats["above"]:,} ({total_above_percent:.2f}%)\n'
                    f'In range: {total_out_of_range_stats["total_elements"] - total_out_of_range_stats["below"] - total_out_of_range_stats["above"]:,} '
                    f'({100 - total_below_percent - total_above_percent:.2f}%)')
    
    plt.figtext(0.02, 0.02, overall_stats, fontsize=9, 
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # 设置总标题
    if region is not None:
        plt.suptitle(f'Tensor Heatmaps for Layer: {target_layer} (Region: [{x_start}:{x_end}, {y_start}:{y_end}])')
    else:
        plt.suptitle(f'Tensor Heatmaps for Layer: {target_layer} (Full Tensor)')
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)  # 为总体统计留出空间
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"热图已保存: {save_path}")
    print(f"总体超出范围统计: {overall_stats}")

# 辅助函数：自动寻找有趣区域
def find_interesting_regions(tensors, region_size=(100, 100), num_regions=5):
    """
    自动寻找数值变化较大的有趣区域
    
    Args:
        tensors: tensor列表
        region_size: 区域大小 (height, width)
        num_regions: 返回的区域数量
    """
    if not tensors:
        return []
    
    # 使用第一个tensor作为参考
    base_tensor = tensors[0]
    if base_tensor.dim() != 2:
        base_tensor = base_tensor.view(base_tensor.size(0), -1)
    
    h, w = base_tensor.shape
    region_h, region_w = region_size
    
    # 计算每个区域的方差（变化程度）
    regions_variance = []
    
    for i in range(0, h - region_h, region_h // 2):
        for j in range(0, w - region_w, region_w // 2):
            region_variances = []
            for tensor in tensors[:10]:  # 只检查前10个步骤以减少计算量
                if tensor.dim() != 2:
                    tensor_2d = tensor.view(tensor.size(0), -1)
                else:
                    tensor_2d = tensor
                
                region = tensor_2d[i:i+region_h, j:j+region_w]
                region_variances.append(region.var().item())
            
            # 使用平均方差作为兴趣度指标
            avg_variance = np.mean(region_variances)
            regions_variance.append((i, j, avg_variance))
    
    # 按方差排序，选择最有趣的区域
    regions_variance.sort(key=lambda x: x[2], reverse=True)
    
    interesting_regions = []
    for i, j, variance in regions_variance[:num_regions]:
        interesting_regions.append({
            'coords': (i, j, region_h, region_w),
            'variance': variance
        })
        print(f"区域 [{i}:{i+region_h}, {j}:{j+region_w}] - 平均方差: {variance:.4f}")
    
    return interesting_regions

def plot_random_points_trend(tensors, valid_steps, save_path, n_points=10):
    """
    随机选择n个点，观察其值在步骤间的变化
    
    Args:
        tensors: 各步骤的tensor列表
        valid_steps: 有效的步骤索引列表
        save_path: 保存路径
        n_points: 要跟踪的点数
    """
    if not tensors:
        print("没有可用的tensor数据")
        return
    
    # 找到所有tensor的共同形状（取第一个tensor的形状）
    base_tensor = tensors[0]
    
    # 计算总元素数
    total_elements = base_tensor.numel()
    
    # 随机选择点
    if total_elements <= n_points:
        # 如果元素数少于n_points，选择所有点
        selected_indices = list(range(total_elements))
        n_points = total_elements
    else:
        selected_indices = np.random.choice(total_elements, n_points, replace=False)
    
    # 获取每个点的坐标（用于图例）
    coordinates = []
    for idx in selected_indices:
        if base_tensor.dim() == 1:
            coord = f"[{idx}]"
        else:
            # 将线性索引转换为多维索引
            coord = np.unravel_index(idx, base_tensor.shape)
            coord_str = "[" + ",".join(map(str, coord)) + "]"
            coordinates.append(coord_str)
    
    # 收集每个点在所有步骤的值
    point_values = np.zeros((n_points, len(tensors)))
    
    for step_idx, tensor in enumerate(tensors):
        flat_tensor = tensor.flatten()
        for point_idx, linear_idx in enumerate(selected_indices):
            if linear_idx < len(flat_tensor):
                point_values[point_idx, step_idx] = flat_tensor[linear_idx].item()
    
    # 绘制趋势图
    plt.figure(figsize=(12, 6))
    
    for point_idx in range(n_points):
        plt.plot(valid_steps, point_values[point_idx, :], 
                marker='o', markersize=2, linewidth=1, 
                label=f'Point {coordinates[point_idx]}' if coordinates else f'Point {selected_indices[point_idx]}')
    
    plt.xlabel('Step')
    plt.ylabel('Value')
    plt.title(f'Random Points Value Trends for Layer: {target_layer}')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"点趋势图已保存: {save_path}")
    
    # 可选：保存数据到CSV
    trend_data = pd.DataFrame(point_values.T, index=valid_steps, 
                             columns=[f'Point_{i}' for i in range(n_points)])
    csv_path = save_path.replace('.png', '.csv')
    trend_data.to_csv(csv_path)
    print(f"点趋势数据已保存: {csv_path}")

# ====== 主执行流程 ======
if __name__ == "__main__":
    # 1. 读取指定层的tensor数据（限制最大步骤数以避免内存问题）
    print(f"开始分析层: {target_layer}")
    max_steps = 50  # 可根据需要调整，或设为None读取所有步骤
    
    tensors, valid_steps = get_layer_tensor(target_layer, files, records_dir, max_steps)
    
    if not tensors:
        print(f"错误: 未能读取到层 '{target_layer}' 的任何有效数据")
        exit(1)
    
    print(f"成功读取 {len(tensors)} 个步骤的数据")
    print(f"Tensor形状示例: {tensors[0].shape}")
    
    # 2. 绘制热图
    heatmap_path = os.path.join(save_dir, f"{target_layer}_heatmaps.png")
    plot_tensor_heatmaps(tensors, valid_steps, heatmap_path, max_plots=25, vmin=-200, vmax=200, region=(0,0,200,200))
    
    # 3. 绘制随机点趋势图
    trend_path = os.path.join(save_dir, f"{target_layer}_point_trends.png")
    plot_random_points_trend(tensors, valid_steps, trend_path, n_points=25)
    
    # 4. 清理内存
    del tensors
    gc.collect()
    torch.cuda.empty_cache()
    
    print("分析完成！")