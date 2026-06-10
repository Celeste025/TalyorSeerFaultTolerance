import json
import pandas as pd
from pathlib import Path
import glob
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import re
from VF_plot import get_v_from_err_prob, get_f_from_err_prob
def find_folders(pattern, search_path=".", recursive=True, return_relative=False):
    """
    使用pathlib的通用文件夹搜索函数
    """
    search_path_obj = Path(search_path)
    if not search_path_obj.exists() or not search_path_obj.is_dir():
        print(f"警告: 搜索路径不存在或不是目录: {search_path}")
        raise ValueError(f"无效的搜索路径: {search_path}")
    
    search_dir = search_path_obj.resolve()
    
    if recursive:
        matches = list(search_dir.rglob(pattern))
    else:
        matches = list(search_dir.glob(pattern))
    
    folders = [f for f in matches if f.is_dir()]
    pattern_re = pattern.replace('*', r'[\d.eE+-]+') + '$'
    filtered_folders = []
    for folder in folders:
        if re.match(pattern_re, folder.name):
            filtered_folders.append(folder)

    folders = filtered_folders
    if return_relative:
        result = [str(f.relative_to(search_dir)) for f in folders]
    else:
        result = [str(f) for f in folders]

    return sorted(result)

def extract_json_from_folder(folder_path, json_filename="run_params.json"):
    """
    从文件夹中提取JSON文件数据
    
    参数:
    folder_path: 文件夹路径
    json_filename: JSON文件名，默认为run_params.json
    
    返回:
    dict: JSON数据，如果文件不存在返回None
    """
    folder_path = Path(folder_path)
    json_file = folder_path / json_filename
    
    if not json_file.exists():
        print(f"警告: {json_filename} 在文件夹 {folder_path} 中不存在")
        return None
    
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 添加文件夹路径信息
        data['folder_path'] = str(folder_path)
        data['folder_name'] = folder_path.name
        
        return data
    except Exception as e:
        print(f"错误: 读取 {json_file} 失败: {e}")
        return None

def folders_to_excel(pattern, search_path=".", output_file="output.xlsx", recursive=True):
    """
    将匹配的文件夹中的JSON数据汇总到Excel文件
    
    参数:
    pattern: 文件夹通配模式
    search_path: 搜索路径
    output_file: 输出Excel文件名
    recursive: 是否递归搜索
    
    返回:
    bool: 操作是否成功
    """
    # 1. 查找匹配的文件夹
    print(f"正在搜索文件夹模式: {pattern}")
    folders = find_folders(pattern, search_path, recursive)
    
    if not folders:
        print("未找到匹配的文件夹")
        return False
    
    print(f"找到 {len(folders)} 个匹配的文件夹")
    
    # 2. 从每个文件夹提取JSON数据
    all_data = []
    for folder in folders:
        # print(f"处理文件夹: {folder}")
        data = extract_json_from_folder(folder)
        if data is not None:
            all_data.append(data)
    
    if not all_data:
        print("未找到有效的JSON数据")
        return False
    
    print(f"成功提取 {len(all_data)} 个JSON文件的数据")
    
    # 3. 转换为DataFrame并保存为Excel
    try:
        df = pd.DataFrame(all_data)
        
        # 重新排列列，让文件夹信息在前
        cols = ['folder_name', 'folder_path'] + [col for col in df.columns if col not in ['folder_name', 'folder_path']]
        df = df[cols]
        
        # 保存为Excel
        df.to_excel(output_file, index=False, engine='openpyxl')
        #print(f"数据已保存到: {output_file}")
        #print(f"Excel包含 {len(df)} 行, {len(df.columns)} 列")
        
        return True
    
    except Exception as e:
        print(f"保存Excel文件失败: {e}")
        return False

def folders_to_dataframe(pattern, search_path=".", recursive=True):
    """
    将匹配的文件夹中的JSON数据汇总到DataFrame（不保存文件）
    
    参数:
    pattern: 文件夹通配模式
    search_path: 搜索路径
    recursive: 是否递归搜索
    
    返回:
    pd.DataFrame: 包含所有数据的DataFrame
    """
    folders = find_folders(pattern, search_path, recursive)
    
    if not folders:
        print("未找到匹配的文件夹")
        return pd.DataFrame()
    
    all_data = []
    for folder in folders:
        data = extract_json_from_folder(folder)
        if data is not None:
            all_data.append(data)
    
    if all_data:
        df = pd.DataFrame(all_data)
        # 重新排列列
        cols = ['folder_name', 'folder_path'] + [col for col in df.columns if col not in ['folder_name', 'folder_path']]
        df = df[cols]
        return df
    else:
        return pd.DataFrame()


def plot_multiple_excel_files(data_files, x_col, y_col, 
                             title="", xlabel="", ylabel="", 
                             save_path=None, figsize=(3.8, 2.8),
                             transparent=True, dpi=700, transfer_vf=False):
    """
    从多个Excel文件读取数据并绘制折线图
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
        'lines.markersize': 7,    # 标记大小
        'mathtext.fontset': 'stix', # 数学字体
        'axes.labelweight': 'bold',      # 坐标轴标签加粗
    })
    
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # 设置透明背景
    if transparent:
        fig.patch.set_alpha(0.0)
        ax.patch.set_alpha(0.0)
    
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    cmap = plt.cm.Set2
    colors = [cmap.colors[i] for i in range(0,8,2)] + [cmap.colors[i] for i in range(1,8,2)] 
    
    for i, (label, file_path) in enumerate(data_files.items()):
        try:
            df = pd.read_excel(file_path)
            
            if transfer_vf and x_col.lower() in ["v", "f"]:
                if 'err_prob' not in df.columns:
                    print(f"警告: 文件 {file_path} 中缺少 err_prob 列, 无法转换 {x_col}")
                    continue
                # 根据 x_col 决定调用哪个函数
                if x_col.lower() == "v":
                    df[x_col] = df['err_prob'].apply(
                        lambda ep: 0.89 if ep <= 1e-7 else get_v_from_err_prob(ep, f=1.0)
                    )

                elif x_col.lower() == "f":
                    df[x_col] = df['err_prob'].apply(
                        lambda ep: 1.02 if ep <= 1e-7 else get_f_from_err_prob(ep, v=0.9)
                    ) * 2
            else:
                if x_col not in df.columns or y_col not in df.columns:
                    print(f"警告: 文件 {file_path} 中缺少列 {x_col} 或 {y_col}, 跳过该文件")
                    continue
            
            if transfer_vf and (x_col.lower() == "f"):
                df = df[df[x_col] <= 4.5]
            df_sorted = df.sort_values(by=x_col)
            marker = markers[i % len(markers)]
            color = colors[i]
            
            ax.plot(df_sorted[x_col], df_sorted[y_col], 
                   marker=marker, label=label, color=color,
                   markeredgecolor='white', markeredgewidth=0.5, 
                   alpha=0.9)
            
        except Exception as e:
            print(f"错误: 处理文件 {file_path} 时出错: {e}")
            continue
    
    # 关键修改：移除 fontsize 参数，让全局设置生效
    ax.set_xlabel(xlabel if xlabel else x_col)  # 移除 fontsize=10
    ax.set_ylabel(ylabel if ylabel else y_col)  # 移除 fontsize=10
    ax.set_title(title, pad=7)  # 移除 fontsize
    ax.xaxis.labelpad = -2
    ax.yaxis.labelpad = -2
    
    # legend = ax.legend(frameon=True, fancybox=False, 
    #                   framealpha=0.8 if not transparent else 0.9,
    #                   edgecolor='gray', loc='best')
    legend = ax.legend(
        # loc='upper left',            # 固定到坐标轴内左上角
        loc='best',
        frameon=True,
        fancybox=True,               # 圆角效果更柔和
        framealpha=0.4,              # 半透明背景（建议 0.3~0.6）
        facecolor='white',           # 白色背景
        edgecolor='gray'
    )
    legend.get_frame().set_linewidth(1)

    ax.grid(True, alpha=0.3, linewidth=1)  # 增加网格线宽
    
    if 'prob' in x_col.lower():
        ax.set_xscale('log')
    
    # 刻度设置 - 使用全局的 xtick.labelsize=15
    if transfer_vf and (x_col.lower() == "v"):
        ax.invert_xaxis()
    ax.tick_params(axis='both', which='major', width=1)  # 增加刻度线宽
    # ax.tick_params(axis='both', which='minor', width=1)
    ax.minorticks_off()
    # 设置边框线宽
    for spine in ax.spines.values():
        spine.set_linewidth(1)  # 增加边框线宽
    
    plt.tight_layout()
    
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
    # #############################
    # # fig1a: bit-scale
    # data_files = {
    #     "bit=8": "target_DiTXL512full_step_50_err_prob_*_bit_8",
    #     "bit=10": "target_DiTXL512full_step_50_err_prob_*_bit_10",
    #     "bit=12": "target_DiTXL512full_step_50_err_prob_*_bit_12",
    #     "bit=16": "target_DiTXL512full_step_50_err_prob_*_bit_16",
    #     "bit=20": "target_DiTXL512full_step_50_err_prob_*_bit_20",
    # }
    # for label, pattern in data_files.items():
    #     output_file = f"plot_code/fig_observation/{label.replace('=','_')}_data.xlsx"
    #     success = folders_to_excel(
    #         pattern=pattern,
    #         search_path="TaylorSeer-DiT/results",
    #         output_file=output_file
    #     )
    #     if not success:
    #         print(f"错误: 生成 {label} 的Excel文件失败")
    #     data_files[label] = output_file
    
    # plot_multiple_excel_files(
    #     data_files=data_files,
    #     x_col="err_prob",
    #     y_col="lpips_score",
    #     xlabel="BER",
    #     ylabel="LPIPS Score",
    #     save_path="plot_code/fig_observation/fig1a.svg"
    # )
    # #############################
    # # fig2a:bit-scale 2
    # data_files = {
    #     "bit=8": "target_PixArtfull_step_30_err_prob_*_bit_8",
    #     "bit=12": "target_PixArtfull_step_30_err_prob_*_bit_12",
    #     "bit=14": "target_PixArtfull_step_30_err_prob_*_bit_14",
    #     "bit=16": "target_PixArtfull_step_30_err_prob_*_bit_16",
    #     "bit=20": "target_PixArtfull_step_30_err_prob_*_bit_20",
    # }
    # for label, pattern in data_files.items():
    #     output_file = f"plot_code/fig_observation/{label.replace('=','_')}_data.xlsx"
    #     success = folders_to_excel(
    #         pattern=pattern,
    #         search_path="PixArt/results",
    #         output_file=output_file
    #     )
    #     if not success:
    #         print(f"错误: 生成 {label} 的Excel文件失败")
    #     data_files[label] = output_file
    
    # plot_multiple_excel_files(
    #     data_files=data_files,
    #     x_col="err_prob",
    #     y_col="lpips_score",
    #     xlabel="BER",
    #     ylabel="LPIPS Score",
    #     save_path="plot_code/fig_observation/fig2a.svg"
    # )


    

    # #############################
    # # fig1b: step-scale
    # data_files = {
    #     "step0-2": "target_DiTXL512full-step0t2_step_50_err_prob_*",
    #     "step9-11": "target_DiTXL512full-step9t11_step_50_err_prob_*",
    #     "step19-21": "target_DiTXL512full-step19t21_step_50_err_prob_*",
    #     "step29-31": "target_DiTXL512full-step29t31_step_50_err_prob_*",
    #     "step39-41": "target_DiTXL512full-step39t41_step_50_err_prob_*",
    #     "step47-49": "target_DiTXL512full-step47t49_step_50_err_prob_*",
    # }
    # for label, pattern in data_files.items():
    #     output_file = f"plot_code/fig_observation/{label.replace('=','_')}_data.xlsx"
    #     success = folders_to_excel(
    #         pattern=pattern,
    #         search_path="TaylorSeer-DiT/results",
    #         output_file=output_file
    #     )
    #     if not success:
    #         print(f"错误: 生成 {label} 的Excel文件失败")
    #     data_files[label] = output_file
    # plot_multiple_excel_files(
    #     data_files=data_files,
    #     x_col="err_prob",
    #     y_col="lpips_score",
    #     xlabel="BER",
    #     ylabel="LPIPS Score",
    #     save_path="plot_code/fig_observation/fig1b.svg"
    # )

    # ######################
    # # fig2b: step-scale 2
    # data_files = {
    #     "step0-2": "target_PixArtfull-step0t2_step_30_err_prob_*",
    #     "step9-11": "target_PixArtfull-step9t11_step_30_err_prob_*",
    #     "step19-21": "target_PixArtfull-step19t21_step_30_err_prob_*",
    #     "step27-29": "target_PixArtfull-step27t29_step_30_err_prob_*",
    # }
    # for label, pattern in data_files.items():
    #     output_file = f"plot_code/fig_observation/{label.replace('=','_')}_data.xlsx"
    #     success = folders_to_excel(
    #         pattern=pattern,
    #         search_path="PixArt/results",
    #         output_file=output_file
    #     )
    #     if not success:
    #         print(f"错误: 生成 {label} 的Excel文件失败")
    #     data_files[label] = output_file
    # plot_multiple_excel_files(
    #     data_files=data_files,
    #     x_col="err_prob",
    #     y_col="lpips_score",
    #     xlabel="BER",
    #     ylabel="LPIPS Score",
    #     save_path="plot_code/fig_observation/fig2b.svg"
    # )

    # ############################
    # # fig1c: module-scale
    # data_files = {
    #     "block0": "target_DiTXL512full_step_50_err_prob_*_layers_0",
    #     "block6": "target_DiTXL512full_step_50_err_prob_*_layers_6",
    #     "block12": "target_DiTXL512full_step_50_err_prob_*_layers_12",
    #     "block18": "target_DiTXL512full_step_50_err_prob_*_layers_18",
    #     "block24": "target_DiTXL512full_step_50_err_prob_*_layers_24",
    #     "embedding": "plot_code/fig_observation/embedding_data.xlsx"
    # }
    # for label, pattern in data_files.items():
    #     if pattern.endswith(".xlsx"):
    #         continue
    #     output_file = f"plot_code/fig_observation/{label.replace('=','_')}_data.xlsx"
    #     if label == "embedding":
    #         search_path = "TaylorSeer-DiT/results"
    #     else:
    #         search_path = "TaylorSeer-DiT/results_layers"
    #     success = folders_to_excel(
    #         pattern=pattern,
    #         search_path=search_path,
    #         output_file=output_file
    #     )
    #     if not success:
    #         print(f"错误: 生成 {label} 的Excel文件失败")
    #     data_files[label] = output_file

    # plot_multiple_excel_files(
    #     data_files=data_files,
    #     x_col="err_prob",
    #     y_col="lpips_score",
    #     xlabel="BER",
    #     ylabel="LPIPS Score",
    #     save_path="plot_code/fig_observation/fig1c.svg"
    # )

    # #####################
    # # fig2c: module-scale 2
    # data_files = {
    #     "block0": "target_PixArtfull_step_30_err_prob_*_layers_0",
    #     "block6": "target_PixArtfull_step_30_err_prob_*_layers_6",
    #     "block12": "target_PixArtfull_step_30_err_prob_*_layers_12",
    #     "block18": "target_PixArtfull_step_30_err_prob_*_layers_18",
    #     "block24": "target_PixArtfull_step_30_err_prob_*_layers_24",
    #     "t-embed": "target_PixArttremb_step_30_err_prob_*",
    #     "cap-embed": "plot_code/fig_observation/cap-embed_data.xlsx"
    # }
    # for label, pattern in data_files.items():
    #     if pattern.endswith(".xlsx"):
    #         continue
    #     output_file = f"plot_code/fig_observation/{label.replace('=','_')}_data.xlsx"
    #     search_path = "PixArt/results_layers" if "embed" not in label else "PixArt/results"
    #     success = folders_to_excel(
    #         pattern=pattern,
    #         search_path=search_path,
    #         output_file=output_file
    #     )
    #     if not success:
    #         print(f"错误: 生成 {label} 的Excel文件失败")
    #     data_files[label] = output_file
    # plot_multiple_excel_files(
    #     data_files=data_files,
    #     x_col="err_prob",
    #     y_col="lpips_score",
    #     xlabel="BER",
    #     ylabel="LPIPS Score",
    #     save_path="plot_code/fig_observation/fig2c.svg"
    # )


    ################################
    # ### Ablation1a: different block_size:
    # data_files = {
    #     "Array size=32": "target_DiTXL512full-step2t_step_50_err_prob_*_protect_ABFT_10_cacheinter_10",
    #     "Array size=64": "target_DiTXL512full-step2t_step_50_err_prob_*_protect_ABFT_10_cacheinter_10_abftblock_64",
    #     "Array size=128": "target_DiTXL512full-step2t_step_50_err_prob_*_protect_ABFT_10_cacheinter_10_abftblock_128",
    #     "Array size=256": "target_DiTXL512full-step2t_step_50_err_prob_*_protect_ABFT_10_cacheinter_10_abftblock_256"
    # }
    # for label, pattern in data_files.items():
    #     output_file = f"plot_code/fig_observation/{label.replace('=','_')}_data.xlsx"
    #     if label == "embedding":
    #         search_path = "TaylorSeer-DiT/results"
    #     else:
    #         search_path = "TaylorSeer-DiT/results_protect"
    #     success = folders_to_excel(
    #         pattern=pattern,
    #         search_path=search_path,
    #         output_file=output_file
    #     )
    #     if not success:
    #         print(f"错误: 生成 {label} 的Excel文件失败")
    #     data_files[label] = output_file
    # plot_multiple_excel_files(
    #     data_files=data_files,
    #     x_col="err_prob",
    #     y_col="lpips_score",
    #     xlabel="BER",
    #     ylabel="LPIPS Score",
    #     save_path="plot_code/fig_observation/ablation_block_size1.svg",
    #     figsize=(2.8, 2.8)
    # )

    # ############################
    # # Ablation1b: different ABFT bit:
    # data_files = {
    #     "ABFT bit=6":"target_DiTXL512full-step2t_step_50_err_prob_*_protect_ABFT_6_cacheinter_10",
    #     "ABFT bit=8":"target_DiTXL512full-step2t_step_50_err_prob_*_protect_ABFT_8_cacheinter_10",
    #     "ABFT bit=10":"target_DiTXL512full-step2t_step_50_err_prob_*_protect_ABFT_10_cacheinter_10",
    #     "ABFT bit=12":"target_DiTXL512full-step2t_step_50_err_prob_*_protect_ABFT_12_cacheinter_10",
    #     "ABFT bit=14":"target_DiTXL512full-step2t_step_50_err_prob_*_protect_ABFT_14_cacheinter_10"
    # }
    # for label, pattern in data_files.items():
    #     output_file = f"plot_code/fig_observation/{label.replace('=','_')}_data.xlsx"
    #     success = folders_to_excel(
    #         pattern=pattern,
    #         search_path="TaylorSeer-DiT/results_protect",
    #         output_file=output_file
    #     )
    #     if not success:
    #         print(f"错误: 生成 {label} 的Excel文件失败")
    #     data_files[label] = output_file
    # plot_multiple_excel_files(
    #     data_files=data_files,
    #     x_col="err_prob",
    #     y_col="lpips_score",
    #     xlabel="BER",
    #     ylabel="LPIPS Score",
    #     save_path="plot_code/fig_observation/DSE_ABFT_bit1.svg",
    #      figsize=(2.8, 2.8)
    # )

    # ################################
    # # Ablation2b: different ABFT bit 2
    # data_files = {
    #     "ABFT bit=8": "target_PixArtfull_step_30_err_prob_*_protect_ABFT_8_cacheinter_8",
    #     "ABFT bit=10": "target_PixArtfull_step_30_err_prob_*_protect_ABFT_10_cacheinter_8",
    #     "ABFT bit=12": "target_PixArtfull_step_30_err_prob_*_protect_ABFT_12_cacheinter_8",
    #     "ABFT bit=14": "target_PixArtfull_step_30_err_prob_*_protect_ABFT_14_cacheinter_8",
    #     "ABFT bit=16": "target_PixArtfull_step_30_err_prob_*_protect_ABFT_16_cacheinter_8",
    #     # "ABFT bit=20": "target_PixArtfull_step_30_err_prob_*_protect_ABFT_20_cacheinter_8"
    # }
    # for label, pattern in data_files.items():
    #     output_file = f"plot_code/fig_observation/{label.replace('=','_')}_data.xlsx"
    #     success = folders_to_excel(
    #         pattern=pattern,
    #         search_path="PixArt/results_protect",
    #         output_file=output_file
    #     )
    #     if not success:
    #         print(f"错误: 生成 {label} 的Excel文件失败")
    #     data_files[label] = output_file
    # plot_multiple_excel_files(
    #     data_files=data_files,
    #     x_col="err_prob",
    #     y_col="lpips_score",
    #     xlabel="BER",
    #     ylabel="LPIPS Score",
    #     save_path="plot_code/fig_observation/DSE_ABFT_bit2.png"
    # )

    # ############################
    # # Ablation1c: different cache interval:
    # data_files = {
    #     # "Interval=1": "target_DiTXL512full-step2t_step_50_err_prob_*_protect_ABFT_10",
    #     "Interval=2": "target_DiTXL512full-step2t_step_50_err_prob_*_protect_ABFT_10_cacheinter_2",
    #     "Interval=5": "target_DiTXL512full-step2t_step_50_err_prob_*_protect_ABFT_10_cacheinter_5",
    #     "Interval=10": "target_DiTXL512full-step2t_step_50_err_prob_*_protect_ABFT_10_cacheinter_10",
    # }
    # for label, pattern in data_files.items():
    #     output_file = f"plot_code/fig_observation/{label.replace('=','_')}_data.xlsx"
    #     success = folders_to_excel(
    #         pattern=pattern,
    #         search_path="TaylorSeer-DiT/results_protect",
    #         output_file=output_file
    #     )
    #     if not success:
    #         print(f"错误: 生成 {label} 的Excel文件失败")
    #     data_files[label] = output_file
    # plot_multiple_excel_files(
    #     data_files=data_files,
    #     x_col="err_prob",
    #     y_col="lpips_score",
    #     xlabel="BER",
    #     ylabel="LPIPS Score",
    #     save_path="plot_code/fig_observation/ablation_cache_interval1.svg",
    #     figsize=(2.8, 2.8)
    # )
    
    

    # ############################
    # Ablation1d: method effectiveness:
    # data_files = {
    #     "No protection":"target_DiTXL512full_step_50_err_prob_*",
    #     "Rollback-ABFT":"target_DiTXL512full_step_50_err_prob_*_protect_ABFT_10_cacheinter_10",
    #     "Rollback-ABFT+\nFine-grained Protection":"target_DiTXL512full-step2t_step_50_err_prob_*_protect_ABFT_10_cacheinter_10"
    # }
    # for label, pattern in data_files.items():
    #     search_path = "TaylorSeer-DiT/results" if label=="No protection" else "TaylorSeer-DiT/results_protect"
    #     output_file = f"plot_code/fig_observation/{label.replace('=','_')}_data.xlsx"
    #     success = folders_to_excel(
    #         pattern=pattern,
    #         search_path=search_path,
    #         output_file=output_file
    #     )
    #     if not success:
    #         print(f"错误: 生成 {label} 的Excel文件失败")
    #     data_files[label] = output_file
    # plot_multiple_excel_files(
    #     data_files=data_files,
    #     x_col="err_prob",
    #     y_col="lpips_score",
    #     xlabel="BER",
    #     ylabel="LPIPS Score",
    #     save_path="plot_code/fig_observation/Ablation_method1.svg",
    #     figsize=(4.8, 2.8)
    # )


    ###################3
    # # Ablation1e: naive method inefficiency:
    # data_files = {
    #     "ThUnderVolt": "target_DiTXL512full-step2t_step_50_err_prob_*_protect_AD_18_cacheinter_10",
    #     "ApproxABFT": "target_DiTXL512full-step2t_step_50_err_prob_*_protect_ABFT_12_cacheorder_-1",
    #     "Ours": "target_DiTXL512full-step2t_step_50_err_prob_*_protect_ABFT_10_cacheinter_10",
    # }
    # for label, pattern in data_files.items():
    #     output_file = f"plot_code/fig_observation/{label.replace('=','_')}_data.xlsx"
    #     success = folders_to_excel(
    #         pattern=pattern,
    #         search_path="TaylorSeer-DiT/results_protect",
    #         output_file=output_file
    #     )
    #     if not success:
    #         print(f"错误: 生成 {label} 的Excel文件失败")
    #     data_files[label] = output_file
    # plot_multiple_excel_files(
    #     data_files=data_files,
    #     x_col="v",
    #     y_col="lpips_score",
    #     xlabel="Voltage (V)",
    #     ylabel="LPIPS Score",
    #     save_path="plot_code/fig_observation/Comparison_naive_methods_v.svg",
    #     transfer_vf=True
    # )
    # plot_multiple_excel_files(
    #     data_files=data_files,
    #     x_col="f",
    #     y_col="lpips_score",
    #     xlabel="Frequency (GHz)",
    #     ylabel="LPIPS Score",
    #     save_path="plot_code/fig_observation/Comparison_naive_methods_f.svg",
    #     transfer_vf=True
    # )

# ##########################
# challenge: poor resilience
    data_files = {
        "DiT-XL-512\nPerformance":"plot_code/fig_observation/No protection_data.xlsx",
    }
    for label, pattern in data_files.items():
        if pattern.endswith(".xlsx"):
            continue
        output_file = f"plot_code/fig_observation/{label.replace('=','_')}_data.xlsx"
        success = folders_to_excel(
            pattern=pattern,
            search_path="TaylorSeer-DiT/results",
            output_file=output_file
        )
        if not success:
            print(f"错误: 生成 {label} 的Excel文件失败")
        data_files[label] = output_file
    plot_multiple_excel_files(
        data_files=data_files,
        x_col="err_prob",
        y_col="image_reward_score",
        xlabel="BER",
        ylabel="ImageReward",
        save_path="plot_code/fig_observation/poor_resilience.svg",
        figsize=(3.3, 2.8)
    )