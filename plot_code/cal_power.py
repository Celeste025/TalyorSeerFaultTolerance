import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math

def generate_abft_excel(
    mem_power: float,
    compute_power: float,
    block_size: int,
    cache_interval: int,
    save_path: str = "abft_analysis.xlsx"
):
    """
    生成包含 ABFT 能耗与错误概率分析的 Excel 文件。
    """
    data = np.array([
        [0.9,   0],
        [0.89,  0],
        [0.88,  0],
        [0.87,  0],
        [0.86,  0],
        [0.85,  0],
        [0.84,  9.58097E-06],
        [0.83,  1.93216E-05],
        [0.82,  3.81642E-05],
        [0.81,  8.37803E-05],
        [0.80,  0.000113002],
        [0.79,  0.000223822],
        [0.78,  0.000226803],
        [0.77,  0.000255173],
        [0.76,  0.00027045],
        [0.75,  0.00028695],
        [0.74,  0.000398888],
        [0.73,  0.000404477],
        [0.72,  0.000450093],
        [0.71,  0.000566715],
        [0.70,  0.000670562],
        [0.69,  0.000810604],
        [0.68,  0.000984711],
        [0.67,  0.001042144],
        [0.66,  0.001146204],
        [0.65,  0.001288854],
        [0.64,  0.001473713],
        [0.63,  0.001630203],
        [0.62,  0.00177775],
        [0.61,  0.001903207],
        [0.60,  0.002171794],
    ])
    # data = np.array([
    #     [0.900, 0],
    #     [0.895, 0],
    #     [0.890, 3.97849E-08],
    #     [0.885, 9.94625E-08],
    #     [0.880, 3.3818E-07],
    #     [0.875, 7.96383E-07],
    #     [0.870, 1.24059E-06],
    #     [0.865, 1.98437E-06],
    #     [0.860, 3.20782E-06],
    #     [0.855, 4.79747E-06],
    #     [0.850, 9.89845E-06],
    #     [0.845, 1.8422E-05],
    #     [0.840, 3.26366E-05],
    #     [0.835, 5.33167E-05],
    #     [0.830, 8.12499E-05],
    #     [0.825, 0.000110752],
    #     [0.820, 0.000153549],
    #     [0.815, 0.000229767],
    #     [0.810, 0.000335342],
    #     [0.805, 0.000456771],
    #     [0.800, 0.000575828],
    #     [0.795, 0.000732837],
    #     [0.790, 0.000907669],
    #     [0.785, 0.001112814],
    #     [0.780, 0.001329086],
    #     [0.775, 0.001616173],
    #     [0.770, 0.001929257],
    #     [0.765, 0.002250735],
    #     [0.760, 0.002574456],
    #     [0.755, 0.002933828],
    #     [0.750, 0.003352878],
    #     [0.745, 0.003776589],
    #     [0.740, 0.004245833],
    #     [0.735, 0.004788337],
    #     [0.730, 0.005413574],
    #     [0.725, 0.006126946],
    #     [0.720, 0.006920478],
    #     [0.715, 0.007746649],
    #     [0.710, 0.008663672],
    #     [0.705, 0.009651201],
    #     [0.700, 0.010704598],
    #     [0.695, 0.011796031],
    #     [0.690, 0.012943897],
    #     [0.685, 0.01402955],
    #     [0.680, 0.015143528],
    #     [0.675, 0.016212043],
    #     [0.670, 0.017292206],
    #     [0.665, 0.018402195],
    #     [0.660, 0.01948775],
    #     [0.655, 0.020597191],
    #     [0.650, 0.021661744],
    #     [0.645, 0.022724465],
    #     [0.640, 0.023803415],
    #     [0.635, 0.024874348],
    #     [0.630, 0.02606078],
    #     [0.625, 0.027167889],
    #     [0.620, 0.028271369],
    #     [0.615, 0.029342815],
    #     [0.610, 0.030388423],
    #     [0.605, 0.031343949],
    # ])

    # data = np.array([
    #     [0.9, 0, 1],
    #     [0.895, 0, 1.0103],
    #     [0.89, 0, 1.021],
    #     [0.885, 0, 1.0318],
    #     [0.88, 0, 1.043],
    #     [0.875, 0, 1.0544],
    #     [0.87, 0, 1.0662],
    #     [0.865, 0, 1.0782],
    #     [0.86, 0, 1.0905],
    #     [0.855, 0, 1.1032],
    #     [0.85, 0, 1.1162],
    #     [0.845, 0, 1.1294],
    #     [0.84, 8.06336E-06, 1.1431],
    #     [0.835, 1.62611E-05, 1.157],
    #     [0.83, 1.62611E-05, 1.1713],
    #     [0.825, 1.62611E-05, 1.1859],
    #     [0.82, 3.21191E-05, 1.2009],
    #     [0.815, 6.27598E-05, 1.2163],
    #     [0.81, 7.05096E-05, 1.232],
    #     [0.805, 9.38486E-05, 1.248],
    #     [0.8, 9.51029E-05, 1.2645],
    #     [0.795, 0.00013103, 1.2812],
    #     [0.79, 0.000188369, 1.2984],
    #     [0.785, 0.000192274, 1.316],
    #     [0.78, 0.000195119, 1.3339],
    #     [0.775, 0.000198124, 1.3522],
    #     [0.77, 0.000224862, 1.3708],
    #     [0.765, 0.000235734, 1.3899],
    #     [0.76, 0.000245496, 1.4093],
    #     [0.755, 0.00025966, 1.4291],
    #     [0.75, 0.000265579, 1.4493],
    #     [0.745, 0.000305488, 1.4699],
    #     [0.74, 0.000375197, 1.4908],
    #     [0.735, 0.000381132, 1.5121],
    #     [0.73, 0.000386757, 1.5338],
    #     [0.725, 0.000394944, 1.5558],
    #     [0.72, 0.000445255, 1.5782],
    #     [0.715, 0.000567444, 1.6009],
    #     [0.71, 0.000608413, 1.6239],
    #     [0.705, 0.000709009, 1.6473],
    #     [0.7, 0.000849336, 1.671],
    #     [0.695, 0.001102127, 1.695],
    #     [0.69, 0.001354376, 1.7193],
    #     [0.685, 0.001805061, 1.7439],
    #     [0.68, 0.00242761, 1.7687],
    #     [0.675, 0.003219272, 1.7938],
    #     [0.67, 0.004021154, 1.8192],
    #     [0.665, 0.005333731, 1.8447],
    #     [0.66, 0.006739287, 1.8705],
    #     [0.655, 0.008886812, 1.8965],
    #     [0.65, 0.010761371, 1.9226],
    #     [0.645, 0.013023623, 1.9488],
    #     [0.64, 0.016265376, 1.9752],
    #     [0.635, 0.019051686, 2.0016],
    #     [0.63, 0.022425558, 2.0281],
    #     [0.625, 0.025976767, 2.0546],
    #     [0.62, 0.029118064, 2.0811],
    #     [0.615, 0.032496016, 2.1076],
    #     [0.61, 0.035907014, 2.1341],
    #     [0.605, 0.041064276, 2.1604],
    #     [0.6, 0.045877635, 2.1866]
    # ])  

    voltages = data[:, 0]
    err_probs = data[:, 1] * 3

    n = block_size ** 2
    err_0 = (1 - err_probs) ** n
    err_1 = (1 - err_probs) ** (n - 1) * err_probs * n
    err_2 = (1 - err_probs) ** (n - 2) * (err_probs ** 2) * (n * (n - 1) / 2)
    err_2p = 1 - err_0 - err_1
    err_3p = 1 - err_0 - err_1 - err_2
    extra_recompute_ratio = err_3p * ( 1 / (1 - err_3p))
    # def get_abft_expected_rows(err_prob, block_size, k_max=6):
    #     """
    #     计算 block_size**2 个数中，出错 >=2 的情况下，
    #     需要修复的行数的数学期望（k 阶近似）。
        
    #     参数:
    #         err_prob: float, 单个元素的出错概率
    #         block_size: int, block 的边长
    #         k_max: int, 近似阶数 (例如 6 表示计算到 err_6)
    #     返回:
    #         abft_expected_rows: float, 数学期望值
    #     """
    #     n = block_size ** 2
    #     err_k = []  # 存储 P(K=k)
    #     for k in range(0, k_max + 1):
    #         comb = math.comb(n, k)
    #         p_k = (err_prob ** k) * ((1 - err_prob) ** (n - k)) * comb
    #         err_k.append(p_k)
    #     weighted_sum = sum(k * err_k[k] for k in range(2, k_max + 1))
    #     return weighted_sum
    
    def get_abft_expected_rows(err_probs, block_size):
        err_probs = err_probs 
        if block_size == 32:
            x , y = 16, 64   # 视为16行 64列 （每次至少读64B）
        elif block_size == 64:
            x , y = 64, 64
        else:
            raise ValueError("Unsupported block size for ABFT expected rows calculation.")
        p_row_clean = (1 - err_probs) ** y
        p_row_dirty = 1 - p_row_clean  ###每行有错误的概率
        expected_rows = p_row_dirty * x  ###期望有多少行出错(不排除1)

        n = x*y
        expected_rows_one = (1 - err_probs) ** (n - 1) * err_probs * n * 1  ###仅有一个数出错的情况的出错行的数学期望
        abft_expected_rows = expected_rows - expected_rows_one  ###排除只有一个数出错的情况
        return abft_expected_rows, x
    
    abft_expected_rows, all_rows = get_abft_expected_rows(err_probs, block_size)
    ########

    abft_extra_mem_ratio = 1 / cache_interval + abft_expected_rows / all_rows 

    low_voltage_compute_power = compute_power * (voltages / 0.9) ** 2 * (1 + 2 / block_size)
    recompute_power = low_voltage_compute_power * (1 + extra_recompute_ratio)
    extra_memory_power = mem_power * (1 + abft_extra_mem_ratio)

    recompute_overall_power = recompute_power + mem_power
    cache_protect_overall_power = low_voltage_compute_power + extra_memory_power

    df = pd.DataFrame({
        "Voltage": voltages,
        "Error_Prob": err_probs,
        "Err_0": err_0,
        "Err_1": err_1,
        "Err_2+": err_2p,
        "Block_Size": block_size,
        "Cache_Interval": cache_interval,
        "ABFT_EXPECTED_Rows": abft_expected_rows,
        "Extra_Recompute_Ratio": extra_recompute_ratio,
        "ABFT_Extra_Mem_Ratio": abft_extra_mem_ratio,
        "ABFT_Expected_Rows": abft_expected_rows,
        "Memory_Power": mem_power,
        "Compute_Power": compute_power,
        "Recompute_Overall_Power": recompute_overall_power,
        "Cache_Protect_Overall_Power": cache_protect_overall_power,
        "Extra_Memory_Power": extra_memory_power,
        "Low_Voltage_Compute_Power": low_voltage_compute_power,
    })

    df.to_excel(save_path, index=False)
    print(f"✅ Excel 文件已保存到: {save_path}")
    return df


def plot_power_vs_voltage(
    df: pd.DataFrame,
    mem_power: float,
    compute_power: float,
    block_size: int,
    cache_interval: int,
    save_fig: str = "power_vs_voltage.png"
):
    """
    绘制功耗随电压变化的曲线。
    包含以下曲线：
        1. Extra_Memory_Power
        2. Memory_Power
        3. Recompute_Overall_Power
        4. Cache_Protect_Overall_Power
        5. Low_Voltage_Compute_Power
    并在图中标注关键参数。
    """
    plt.figure(figsize=(8, 6))

    # 绘制各功耗曲线
    plt.plot(df["Voltage"], df["Extra_Memory_Power"], label="Extra_Memory_Power", marker='o')
    plt.plot(df["Voltage"], df["Memory_Power"], label="Memory_Power", marker='s')
    plt.plot(df["Voltage"], df["Recompute_Overall_Power"], label="Recompute_Overall_Power", marker='^')
    plt.plot(df["Voltage"], df["Cache_Protect_Overall_Power"], label="Cache_Protect_Overall_Power", marker='x')
    plt.plot(df["Voltage"], df["Low_Voltage_Compute_Power"], label="Low_Voltage_Compute_Power", marker='d')

    # 坐标轴设置
    plt.xlabel("Voltage (V)", fontsize=12)
    plt.ylabel("Power (W)", fontsize=12)
    plt.title("Power Consumption vs Voltage under ABFT Protection", fontsize=13)
    plt.ylim(bottom=0, top=11)  # ✅ 纵轴从 0 开始

    # 添加网格和图例
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()

    # 在右上角添加参数标注
    textstr = (
        f"Memory Power = {mem_power}\n"
        f"Compute Power = {compute_power}\n"
        f"Block Size = {block_size}\n"
        f"Cache Interval = {cache_interval}"
    )
    # 添加文本框（位置在图的右上角）
    plt.text(
        0.98, 0.95, textstr,
        transform=plt.gca().transAxes,  # 相对坐标
        fontsize=10,
        verticalalignment='top',
        horizontalalignment='right',
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.6, edgecolor='gray')
    )

    plt.tight_layout()
    plt.savefig(save_fig, dpi=300)
    print(f"✅ 图像已保存为: {save_fig}")
    plt.close()


# ====================== 示例运行 ======================
if __name__ == "__main__":
    mem_power = 1
    compute_power = 1152 * 0.16 / 31
    block_size = 32
    cache_interval = 10
    save_path = f"power_analysis_mem{mem_power}_com{compute_power}_block{block_size}_inter{cache_interval}.xlsx"

    df = generate_abft_excel(mem_power, compute_power, block_size, cache_interval, save_path = save_path)
    
    save_fig = f"fig_mem{mem_power}_com{compute_power}_block{block_size}_inter{cache_interval}.png"
    plot_power_vs_voltage(df, mem_power, compute_power, block_size, cache_interval, save_fig = save_fig)

#### 0.16pJ/B  31pJ/B

