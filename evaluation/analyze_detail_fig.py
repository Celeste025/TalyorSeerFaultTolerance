import matplotlib.pyplot as plt
import numpy as np
import os
import re

def load_scores_from_txt(txt_file):
    """
    从txt文件加载分数数据
    """
    scores = {}
    with open(txt_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if ':' in line:
                filename, score = line.split(':', 1)
                scores[filename.strip()] = float(score.strip())
    return scores

def analyze_multiple_txt_files(txt_files, labels=None):
    """
    分析多个txt文件的分数差异
    """
    if labels is None:
        labels = [f"File_{i+1}" for i in range(len(txt_files))]
    
    all_scores = []
    filenames = None
    
    # 加载所有文件的分数
    for txt_file in txt_files:
        scores = load_scores_from_txt(txt_file)
        all_scores.append(scores)
        
        # 确保所有文件有相同的图片名称
        if filenames is None:
            filenames = list(scores.keys())
        else:
            # 检查文件名是否一致
            if set(filenames) != set(scores.keys()):
                print("警告: 不同文件中的图片名称不一致！")
    
    # 按图片名称排序，确保顺序一致
    filenames.sort()
    
    # 提取图片索引用于横轴
    indices = list(range(len(filenames)))
    
    return filenames, all_scores, indices, labels

def plot_score_comparison(txt_files, labels=None, save_path=None):
    """
    绘制多个txt文件的分数对比图
    """
    filenames, all_scores, indices, labels = analyze_multiple_txt_files(txt_files, labels)
    
    # 创建图表
    plt.figure(figsize=(15, 8))
    
    # 为每个文件绘制分数曲线
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
    markers = ['o', 's', '^', 'D', 'v', '<']
    
    for i, (scores_dict, label) in enumerate(zip(all_scores, labels)):
        # 按排序后的文件名获取分数
        scores = [scores_dict[filename] for filename in filenames]
        
        plt.plot(indices, scores, 
                color=colors[i % len(colors)],
                marker=markers[i % len(markers)],
                markersize=4,
                linewidth=1,
                alpha=0.7,
                label=label)
    
    plt.xlabel('Image Index (0-49)', fontsize=12)
    plt.ylabel('CLIP Score', fontsize=12)
    plt.title('CLIP Score Comparison Across Multiple Evaluations', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 设置横轴刻度
    plt.xticks(range(0, len(filenames), 5))
    
    # 自动调整布局
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存至: {save_path}")
    
    return filenames, all_scores

def plot_score_differences(txt_files, labels=None, save_path=None):
    """
    绘制分数差异图
    """
    filenames, all_scores, indices, labels = analyze_multiple_txt_files(txt_files, labels)
    
    if len(all_scores) != 2:
        print("差异图仅支持两个文件的比较")
        return
    
    # 计算差异
    scores1 = [all_scores[0][filename] for filename in filenames]
    scores2 = [all_scores[1][filename] for filename in filenames]
    differences = [abs(s1 - s2) for s1, s2 in zip(scores1, scores2)]
    
    # 创建差异图
    plt.figure(figsize=(15, 8))
    
    # 绘制差异柱状图
    bars = plt.bar(indices, differences, alpha=0.7, color='red')
    
    # 标记差异较大的点
    threshold = 0.3  # 差异阈值
    large_diff_indices = [i for i, diff in enumerate(differences) if diff > threshold]
    for idx in large_diff_indices:
        bars[idx].set_color('darkred')
        plt.annotate(f'{differences[idx]:.3f}', 
                    xy=(idx, differences[idx]),
                    xytext=(0, 5),
                    textcoords='offset points',
                    ha='center', va='bottom',
                    fontweight='bold')
    
    plt.xlabel('Image Index (0-49)', fontsize=12)
    plt.ylabel('Score Difference', fontsize=12)
    plt.title(f'CLIP Score Differences Between {labels[0]} and {labels[1]}', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # 设置横轴刻度
    plt.xticks(range(0, len(filenames), 5))
    
    # 添加阈值线
    plt.axhline(y=threshold, color='orange', linestyle='--', alpha=0.8, 
                label=f'Large Difference Threshold ({threshold})')
    plt.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"差异图已保存至: {save_path}")

    
    # 打印差异最大的图片
    print("\n差异最大的前10张图片:")
    print("=" * 80)
    diff_data = [(filenames[i], scores1[i], scores2[i], differences[i]) 
                for i in range(len(filenames))]
    diff_data.sort(key=lambda x: x[3], reverse=True)
    
    for i, (filename, score1, score2, diff) in enumerate(diff_data[:10]):
        print(f"{i+1:2d}. {filename:<35} {score1:8.4f} -> {score2:8.4f} | 差异: {diff:.4f}")

# 使用示例
if __name__ == "__main__":
    # 指定你的txt文件路径
    txt_files = [
        "/data/home/jinqiwen/workspace/diffusion_fault_tolerance/TaylorSeerFaultTolerance/TaylorSeer-DiT/results_taylorseer/target_DiTXL512full-step2t_step_50_err_prob_0.0_protect_ABFT_12_cacheinter_9_tinter_3_torder_2/images_gen/image_reward_scores.txt",  
        "/data/home/jinqiwen/workspace/diffusion_fault_tolerance/TaylorSeerFaultTolerance/TaylorSeer-DiT/results_taylorseer/target_DiTXL512full-step2t_step_50_err_prob_0.003_protect_ABFT_12_cacheinter_9_tinter_3_torder_2/images_gen/image_reward_scores.txt",  
        "/data/home/jinqiwen/workspace/diffusion_fault_tolerance/TaylorSeerFaultTolerance/TaylorSeer-DiT/results_taylorseer/target_DiTXL512full-step2t_step_50_err_prob_0.001_protect_ABFT_12_cacheinter_9_tinter_3_torder_2/images_gen/image_reward_scores.txt",  
    ]
    
    # 对应的标签
    labels = ["err0", "err1e-3", "err3e-3"]
    
    # 检查文件是否存在
    for file in txt_files:
        if not os.path.exists(file):
            print(f"文件不存在: {file}")
        else:
            print(f"找到文件: {file}")
    
    # 绘制对比图
    print("绘制分数对比图...")
    filenames, all_scores = plot_score_comparison(
        txt_files, 
        labels=labels,
        save_path="score_comparison.png"
    )
    
    # 绘制差异图
    print("\n绘制分数差异图...")
    plot_score_differences(
        txt_files,
        labels=labels,
        save_path="score_differences.png"
    )
    
    # 打印统计信息
    print("\n统计信息:")
    print("=" * 50)
    for i, (label, scores_dict) in enumerate(zip(labels, all_scores)):
        scores = list(scores_dict.values())
        print(f"{label}:")
        print(f"  平均分: {np.mean(scores):.4f}")
        print(f"  标准差: {np.std(scores):.4f}")
        print(f"  最高分: {np.max(scores):.4f}")
        print(f"  最低分: {np.min(scores):.4f}")
