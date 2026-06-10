import pandas as pd
import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.optimize import root_scalar

def read_and_scale_excel(file_path, save_path, scale=1, sheet_name=2):
    """
    读取 Excel 文件，将 VDD/f 数据矩阵提取为 numpy 数组，
    将 X * scale 后保存到新的 Excel 文件。
    
    Args:
        file_path: 原 Excel 文件路径
        save_path: 保存 Excel 文件路径
        scale: 缩放系数，默认 3
    
    Returns:
        VDD: 1D numpy array, shape (num_vdd,)
        F: 1D numpy array, shape (num_f,)
        X_scaled: 2D numpy array, shape (num_vdd, num_f), X*scale
    """
    # 读取原始 Excel
    df = pd.read_excel(file_path, header=None, sheet_name=sheet_name)
    
    # 第一行作为 F
    F = df.iloc[0, 1:].to_numpy(dtype=float)
    # 第一列作为 VDD
    VDD = df.iloc[1:, 0].to_numpy(dtype=float)
    # 剩余的矩阵是数据
    X = df.iloc[1:, 1:].to_numpy(dtype=float)
    
    # X * scale
    X_scaled = X * scale
    
    # 保存到新 Excel
    # 构建新的 DataFrame，包括第一行第一列
    df_scaled = pd.DataFrame(np.zeros_like(df, dtype=float))
    df_scaled.iloc[0, 1:] = F
    df_scaled.iloc[1:, 0] = VDD
    df_scaled.iloc[1:, 1:] = X_scaled
    
    # 保存
    df_scaled.to_excel(save_path, header=False, index=False)
    print(f"Scaled data saved to {save_path}")
    return VDD, F, X_scaled

def read_vdd_f_excel(file_path, sheet_name=2):
    """
    读取 Excel 文件，将 VDD/f 数据矩阵提取为 numpy 数组。
    假设第一列是 VDD，第一行是频率 f（左上角无效元素忽略）。
    
    Args:
        file_path: Excel 文件路径
        sheet_name: 工作表名称或索引，默认为2（第三个工作表）
    
    Returns:
        VDD: 1D numpy array, shape (num_vdd,)
        F: 1D numpy array, shape (num_f,)
        X: 2D numpy array, shape (num_vdd, num_f), 对应每个 (VDD, F) 的数值
    """
    df = pd.read_excel(file_path, header=None, sheet_name=sheet_name)
    # 第一行作为 F
    F = df.iloc[0, 1:].to_numpy(dtype=float)
    # 第一列作为 VDD
    VDD = df.iloc[1:, 0].to_numpy(dtype=float)
    # 剩余的矩阵是数据
    X = df.iloc[1:, 1:].to_numpy(dtype=float)
    return VDD, F, X

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

def plot_isolines(VDD, F, X, x_targets=[0.001], method='linear', save_path='isoline_plot.svg', plot=True):
    """
    绘制多个等值线，横轴 VDD，纵轴 F。
    
    Args:
        VDD: 1D array of VDD values
        F: 1D array of frequency values
        X: 2D array of shape (len(VDD), len(F))
        x_targets: list of 等值线目标值
        method: 插值方法 'linear', 'cubic', 'nearest'
        save_path: 保存图片路径
        
    Returns:
        contours_dict: dict, key=x_target, value=numpy array of shape (N,2) 每行是 (VDD,F)
    """
    # 设置科研论文风格的参数
    plt.rcParams.update({
        'font.family': 'Liberation Sans',
        'font.size': 12,
        'axes.titlesize': 12,
        'axes.labelsize': 12,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'lines.linewidth': 4,
        'mathtext.fontset': 'custom',  # 关键：使用自定义字体
        'mathtext.rm': 'Liberation Sans',      # 常规字体
        'mathtext.it': 'Liberation Sans:italic', # 斜体
        'mathtext.bf': 'Liberation Sans:bold',   # 粗体
        'mathtext.sf': 'Liberation Sans',        # 无衬线
        'mathtext.tt': 'Liberation Sans',        # 等宽
        'axes.labelweight': 'bold',
    })
    F *= 2
    x_targets = sorted(x_targets)
    # 创建二维网格
    F_grid, VDD_grid = np.meshgrid(F, VDD)
    points = np.column_stack((VDD_grid.ravel(), F_grid.ravel()))
    values = X.ravel()
    
    # 创建细网格用于平滑绘图
    F_fine = np.linspace(F.min(), F.max(), 500)
    VDD_fine = np.linspace(VDD.min(), VDD.max(), 500)
    F_fine_grid, VDD_fine_grid = np.meshgrid(F_fine, VDD_fine)
    
    # 插值
    X_fine = griddata(points, values, (VDD_fine_grid, F_fine_grid), method=method)
    
    # 绘制等值线
    fig, ax = plt.subplots(figsize=(4.3, 2.8))
    
    # 设置透明背景
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)
    
    colors = plt.cm.Set2(np.linspace(0, 1, 10))
    
    # 检查是否有0线需要平移
    has_zero = 0 in x_targets
    contour_levels = []
    contour_colors = []

    # 分离0线和其他线
    for i, target in enumerate(x_targets):
        if target == 0 and has_zero:
            # 0线单独处理
            continue
        contour_levels.append(target)
        contour_colors.append(colors[i])

    CS = None  # 初始化 CS 变量

    # 绘制非0等值线
    if contour_levels:
        CS = ax.contour(VDD_fine_grid, F_fine_grid, X_fine, levels=contour_levels, colors=contour_colors, linewidths=2)

    # 处理0线
    if has_zero:
        # 计算0线的等值线但不显示在当前图形上
        zero_fig, zero_ax = plt.subplots(figsize=(6, 5))
        zero_CS = zero_ax.contour(VDD_fine_grid, F_fine_grid, X_fine, levels=[0])
        plt.close(zero_fig)  # 关闭临时图形
        
        # 获取0线的坐标点并向右平移0.1
        for path in zero_CS.get_paths():
            vertices = path.vertices
            if len(vertices) > 0:
                # 向右平移0.1，确保不超出坐标范围
                new_vertices = vertices.copy()
                new_vertices[:, 0] += 0.04
                
                # 检查是否超出VDD范围，只保留范围内的点
                vdd_min, vdd_max = VDD.min(), VDD.max()
                # 找出在范围内的点
                valid_mask = (new_vertices[:, 0] >= vdd_min) & (new_vertices[:, 0] <= vdd_max)
                valid_vertices = new_vertices[valid_mask]
                
                # 只有在有效点足够多时才绘制
                if len(valid_vertices) > 1:
                    # 绘制平移后的0线
                    ax.plot(valid_vertices[:, 0], valid_vertices[:, 1], '--', color=colors[7], linewidth=2, alpha=1)
    
    # 在v=0.9, f=1.0处画两条线
    v_ref, f_ref = 0.9, 1.0*2
    
    # 垂直线 (VDD=0.9)
    ax.axvline(x=v_ref, color='gray', linewidth=1, linestyle=':', alpha=1)
    
    # 水平线 (F=1.0)
    ax.axhline(y=f_ref, color='gray', linewidth=1, linestyle=':', alpha=1)
    
    # 在交点处打点
    ax.plot(v_ref, f_ref, marker='o', color=colors[-1], markersize=10, markeredgecolor='white', markeredgewidth=1, label='Standard Working V-f')
    
    # 设置标签和标题
    ax.set_xlabel("VDD(V)")
    ax.set_ylabel("Frequency(GHz)")

    ax.xaxis.labelpad = -2
    ax.yaxis.labelpad = -2
    
    # 在图例中显示等值线标签（左上角）
    from matplotlib.lines import Line2D
    legend_elements = []
    for i, target in enumerate(x_targets):
        if target == 0:
            # 0线用虚线表示
            legend_elements.append(Line2D([0], [0], color=colors[7], lw=2, label=f'BER=0'))
        else:
            mantissa = target / (10**np.floor(np.log10(target)))
            exp = int(np.floor(np.log10(target)))

            legend_elements.append(Line2D([0], [0], color=colors[i], lw=2, 
                            label = rf'$\mathrm{{BER}}={mantissa:g}\times 10^{{{exp}}}$'))
    
    legend_elements.append(Line2D([0], [0], marker='o', color=colors[-1], markeredgecolor='white', 
                                markersize=8, lw=0, label='Nominal V-f'))
    
    # 设置图例在左上角
    legend = ax.legend(handles=legend_elements, 
                      loc='upper left',
                      frameon=True,
                      fancybox=True,
                      framealpha=0.4,
                      facecolor='white',
                      edgecolor='gray',
                      labelspacing=0.2)
    legend.get_frame().set_linewidth(1)
    
    # 网格和边框设置
    ax.grid(True, alpha=0.3, linewidth=1)
    ax.tick_params(axis='both', which='major', width=1)
    
    # 设置边框线宽
    for spine in ax.spines.values():
        spine.set_linewidth(1)
    
    custom_vdd_ticks = [0.6, 0.7, 0.8, 0.9, 1.0, 1.1]  # 你可以修改这个列表
    ax.set_xticks(custom_vdd_ticks)
    plt.tight_layout()
    
    if plot:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                   transparent=True,
                   facecolor='none',
                   edgecolor='none')
        print(f"等值线图已保存到: {save_path}")
    
    plt.close(fig)
    
    # 提取等值线点
    contours_dict = {}
    try:
        if CS is not None:
            for idx, target in enumerate(contour_levels):
                if hasattr(CS, 'allsegs') and idx < len(CS.allsegs):
                    level_segs = CS.allsegs[idx]
                    if level_segs:
                        contours_dict[target] = np.vstack(level_segs)
                    else:
                        contours_dict[target] = np.empty((0,2))
        
        # 添加0线到返回结果
        if has_zero:
            contours_dict[0] = new_vertices if 'new_vertices' in locals() else np.empty((0,2))
            
    except Exception as e:
        print("Failed to extract contour points:", e)
        for target in x_targets:
            contours_dict[target] = np.empty((0,2))
    
    return contours_dict

###### 提供一个快捷计算X = X(VDD, F)的函数
import numpy as np
from scipy.interpolate import griddata

def check_vdd_f_x_shapes(VDD, F, X):
    """
    检查 VDD, F, X 的形状是否满足条件
    """
    n = len(VDD)
    m = len(F)
    
    # 检查 X 的形状是否为 n x m
    if isinstance(X, list):
        if len(X) != n:
            raise ValueError(f"X 的行数 ({len(X)}) 与 VDD 的长度 ({n}) 不匹配")
        for i, row in enumerate(X):
            if len(row) != m:
                raise ValueError(f"X 的第 {i} 行列数 ({len(row)}) 与 F 的长度 ({m}) 不匹配")
    elif isinstance(X, np.ndarray):
        if X.shape != (n, m):
            raise ValueError(f"X 的形状 {X.shape} 与预期的 ({n}, {m}) 不匹配")
    else:
        raise TypeError("X 必须是列表或 numpy 数组")
    
    # print(f"形状检查通过: VDD({n}), F({m}), X({n}x{m})")
    return True

def get_ber_value(v, f, method='linear'):
    """
    根据 VDD, F, X 数据，通过插值获取指定 v, f 对应的 x 值
    
    参数:
        v: 指定的VDD值
        f: 指定的F值
        method: 插值方法 ('linear', 'cubic', 'nearest')
    
    返回:
        interpolated_x: 插值得到的x值
    """
    VDD, F, X = read_vdd_f_excel('/data/home/jinqiwen/workspace/diffusion_fault_tolerance/TaylorSeerFaultTolerance/plot_code/VFcurve.xlsx')  # 读取你的 excel
    # 检查形状
    check_vdd_f_x_shapes(VDD, F, X)
    
    # 将数据转换为 numpy 数组
    VDD_arr = np.array(VDD)
    F_arr = np.array(F)
    X_arr = np.array(X)
    
    # 创建网格点
    v_grid, f_grid = np.meshgrid(VDD_arr, F_arr, indexing='ij')
    
    # 准备插值数据点
    points = np.column_stack((v_grid.ravel(), f_grid.ravel()))
    values = X_arr.ravel()
    
    # 执行二维插值
    interpolated_x = griddata(points, values, (v, f), method=method)
    
    # 如果插值结果为 nan（在数据范围外），使用最近邻方法
    if np.isnan(interpolated_x):
        print(f"警告: ({v}, {f}) 在数据范围外，使用最近邻插值")
        interpolated_x = griddata(points, values, (v, f), method='nearest')
    
    return interpolated_x

from scipy.interpolate import interp1d

def get_v_from_err_prob(err_prob, f, VDD=None, F=None, X=None):
    """已知 f, err_prob 求 v —— 使用对应列的一维线性插值"""
    if VDD is None or F is None or X is None:
        VDD, F, X = read_vdd_f_excel(
            '/data/home/jinqiwen/workspace/diffusion_fault_tolerance/TaylorSeerFaultTolerance/plot_code/VFcurve.xlsx'
        )

    # 找到 f 对应的列索引
    f_idx = (np.abs(F - f)).argmin()
    
    # 该列的数据（随 VDD 变化）
    err_col = X[:, f_idx]

    # 构造插值函数（err_prob → v）
    interp_func = interp1d(err_col, VDD, kind='linear', fill_value="extrapolate")

    return float(interp_func(err_prob))


def get_f_from_err_prob(err_prob, v, VDD=None, F=None, X=None):
    """已知 v, err_prob 求 f —— 使用对应行的一维线性插值"""
    if VDD is None or F is None or X is None:
        VDD, F, X = read_vdd_f_excel(
            '/data/home/jinqiwen/workspace/diffusion_fault_tolerance/TaylorSeerFaultTolerance/plot_code/VFcurve.xlsx'
        )

    # 找到 VDD 对应的行索引
    v_idx = (np.abs(VDD - v)).argmin()

    # 该行的数据（随 F 变化）
    err_row = X[v_idx, :]

    # 构造插值函数（err_prob → f）
    interp_func = interp1d(err_row, F, kind='linear', fill_value="extrapolate")

    return float(interp_func(err_prob))

if __name__ == "__main__":
    #VDD, F, X = read_and_scale_excel('VFcurve.xlsx', 'VFcurve_scaled.xlsx', scale=3, sheet_name=2)
    VDD, F, X = read_vdd_f_excel('/data/home/jinqiwen/workspace/diffusion_fault_tolerance/TaylorSeerFaultTolerance/plot_code/VFcurve.xlsx')  # 读取你的 excel
    # b = get_ber_value(v=0.6, f=1.0)
    # print(f"Interpolated BER at VDD=0.6V, F=1.0GHz: {b}")
    # v = get_v_from_err_prob(0.003, f=1.0, VDD=VDD, F=F, X=X)
    # print(f"VDD for BER=0.003 at F=1.0GHz: {v}")
    # f = get_f_from_err_prob(0.003, v=0.9, VDD=VDD, F=F, X=X)
    # print(f"F for BER=0.003 at VDD=0.9V: {f}")
    # v = get_v_from_err_prob(1e-8, f=1.0, VDD=VDD, F=F, X=X)
    # print(f"VDD for BER=1e-8 at F=1.0GHz: {v}")
    result = plot_isolines(VDD, F, X, x_targets=[ 1e-3, 1e-4, 1e-6,  0], plot=True,save_path='challenge_a.svg')    # 画等值线
  
    # # print(result[0.003][0:5])  [(V,f)pairs]
    # for (v,f) in result[0.003]:
    #     print(f"{v:.4f}, {f:.4f}")
    #     print( (v/0.9)**2 )
    # print(get_interpolated_value(0.85, 1.2))
