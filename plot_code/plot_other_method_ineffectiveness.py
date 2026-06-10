import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
def blend_color(c1, c2, alpha):
    """
    c1, c2: RGB tuple，范围 0~1
    alpha: 叠加透明度 0~1（c2的透明度）
    返回混合后的 RGB
    """
    return tuple((1 - alpha) * np.array(c1) + alpha * np.array(c2))

def plot_grouped_bars_from_excel(
    data_file,
    x_col,
    plot_columns=None,       # 指定绘制列
    title="",
    xlabel="",
    ylabel_left="",
    figsize=(5, 3.2),
    dpi=500,
    save_path=None,
    transparent=True,
    bar_width=0.3,
    rotation=0,
    extra_hatch='/////////////',       # Extra的花纹
    extra_alpha=1.0,           # Extra花纹的透明度
    y_scale = 1000
):
    """
    绘制带 Base/Extra 分段的分组柱状图。
    Base: 纯色填充
    Extra (Protection Overhead): 在Base上覆盖纹理表示
    """
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from pathlib import Path
    from matplotlib.patches import Patch

    # -------------------- Style --------------------
    plt.rcParams.update({
        'font.family': 'Liberation Sans',
        'font.size': 12,
        'axes.titlesize': 12,
        'axes.labelsize': 12,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 11,
        'lines.linewidth': 2,
        'lines.markersize': 6,
        'mathtext.fontset': 'stix',
        'axes.labelweight': 'bold',
    })

    # -------------------- Read data --------------------
    data_file = Path(data_file)
    if not data_file.exists():
        raise FileNotFoundError(f"Data file not found: {data_file}")

    df = pd.read_excel(data_file) if data_file.suffix.lower() in ['.xls', '.xlsx'] else pd.read_csv(data_file)
    if x_col not in df.columns:
        raise ValueError(f"{x_col} not found in columns of {data_file}")

    all_cols = list(df.columns)

    # -------------------- Detect methods --------------------
    if plot_columns is not None:
        method_cols = [c for c in plot_columns if c in df.columns]
        if len(method_cols) == 0:
            raise ValueError(f"No columns found in plot_columns: {plot_columns}")
    else:
        method_cols = [c for c in all_cols if c != x_col and not c.endswith("Base")]

    base_cols = {c[:-4]: c for c in all_cols if c.endswith("Base")}

    x = np.arange(len(df))
    n_methods = len(method_cols)
    total_bar_width = bar_width
    single_bar_width = total_bar_width / n_methods
    offsets = (np.arange(n_methods) - (n_methods - 1) / 2.0) * single_bar_width

    cmap = plt.cm.Set2
    # base_colors = cmap(np.linspace(0, 0.75, n_methods))
    base_colors = cmap([0, 0.5, 0.25])


    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    if transparent:
        fig.patch.set_alpha(0.0)
        ax.patch.set_alpha(0.0)

    extra_legend_added = False
    
    # ----------------- scale y as real power
    for col in method_cols:
        df[col] = df[col].astype(float) * y_scale
    for base_key, base_col in base_cols.items():
        df[base_col] = df[base_col].astype(float) * y_scale


    df = df.iloc[::-1].reset_index(drop=True)
    x = np.arange(len(df))
    # -------------------- Draw bars --------------------
    for i, method in enumerate(method_cols):
        vals = df[method].values.astype(float)
        if method in base_cols:
            base_vals = df[base_cols[method]].values.astype(float)
            extra_vals = np.clip(vals - base_vals, 0.0, None)
        else:
            base_vals = vals.copy()
            extra_vals = np.zeros_like(vals)

        # 先画 Base
        base_bar = ax.bar(
            x + offsets[i],
            base_vals,
            width=single_bar_width * 0.93,
            color=base_colors[i],
            edgecolor='none',
            label=method
        )

        # Extra：保持原本颜色为底色 + 斜线
        if np.any(extra_vals > 0):
            gray = (0.5, 0.5, 0.5, 1.0)
            alpha = 0.6   # 你想要的灰色透明度
            blended_color = blend_color(base_colors[i], gray, alpha)
            ax.bar(
                x + offsets[i],
                extra_vals,
                width=single_bar_width * 0.93,
                bottom=base_vals,
                # color=base_colors[i],   # 使用 Base 的颜色
                color = blended_color,
                edgecolor='gray',      # 斜线颜色
                linewidth=0,            # 边框可不要
                alpha = 1
            )
            extra_legend_added = True

    y_max = 1.57 * df[method_cols].max().max()
    ax.set_ylim(0, y_max)

    # -------------------- Legend --------------------
    handles, labels = ax.get_legend_handles_labels()
    if extra_legend_added:
        extra_patch = Patch(facecolor='none', edgecolor='gray',
                        label='Recovery Overhead')
        handles.append(extra_patch)
        labels.append('Recovery Overhead')

    ax.legend(handles, labels, frameon=True, fancybox=True, framealpha=0.4,
              edgecolor='gray', loc='upper left', labelspacing=0.2).get_frame().set_linewidth(1)

    # -------------------- Axis, grid, labels --------------------
    ax.set_xticks(x)
    ax.set_xticklabels(df[x_col].astype(str).values, rotation=rotation)
    ax.set_xlabel(xlabel if xlabel else x_col)
    ax.set_ylabel(ylabel_left if ylabel_left else "Value")
    ax.set_title(title, pad=7)

    ax.grid(True, alpha=0.25, linewidth=0.8)
    for spine in ax.spines.values():
        spine.set_linewidth(1)

    plt.tight_layout()
    if save_path:
        import os
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight',
                    transparent=transparent, facecolor='none' if transparent else 'white')
        print(f"Saved grouped bar plot to: {save_path}")

    plt.close(fig)




import pandas as pd
import numpy as np
from VF_plot import read_vdd_f_excel   # 你已有
from VF_plot import get_ber_value    # 你已有
from pathlib import Path

###############################################
# 计算 recompute、ABFT 行访问比例 等逻辑
###############################################
def get_abft_expected_rows(err_prob, block_size):
    """
    返回 (abft_expected_rows, x)
    """
    if block_size == 32:
        x, y = 16, 64
    elif block_size == 64:
        x, y = 64, 64
    else:
        raise ValueError("Unsupported block size.")

    p_row_clean = (1 - err_prob) ** y
    p_row_dirty = 1 - p_row_clean             # 每行出错概率
    expected_rows = p_row_dirty * x           # 出错行期望

    n = x * y
    expected_rows_one = (1 - err_prob) ** (n - 1) * err_prob * n * 1

    abft_expected_rows = expected_rows - expected_rows_one
    return abft_expected_rows, x


###############################################
# 生成最终 Excel 文件
###############################################
def generate_barplot_excel(
    vf_curve_path="VFcurve.xlsx",
    output_excel="barplot_data.xlsx",
    k=7,
    block_size=64,
    freq_for_err=1.0,
):
    """
    生成柱状图 Excel 文件，包含：

    V, err_prob,
    DMRBase, OursBase, ABFTRecomputeBase,
    recompute_prob, ABFTRecompute, line_access_ratio,
    Ours
    """

    # =============================
    # 1. 读取 VF 曲线矩阵
    # =============================
    VDD, F, X = read_vdd_f_excel(vf_curve_path)

    # 你希望的 V 列
    V_list = np.array([0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9])
    # --------------------------
    # 2. 计算 err_prob
    # --------------------------
    err_prob_list = np.array([get_ber_value(v=v, f=freq_for_err) for v in V_list])

    # =============================
    # 3. 计算各 base 值
    # =============================
    # DMR: 1 + k*(v/0.9)^2 * 3
    DMRBase = 1 + k * (V_list / 0.9) ** 2 * 2

    # OursBase: 1.1 + k*(v/0.9)^2 * (1 + 1/32)
    OursBase = 1.1 + k * (V_list / 0.9) ** 2 * (1 + 1/32)

    # ABFTRecomputeBase: 1 + k*(v/0.9)^2 * (1 + 1/32)
    ABFTRecomputeBase = 1 + k * (V_list / 0.9) ** 2 * (1 + 1/32)

    # =============================
    # 4. recompute_prob
    #  recompute_prob = 1 - (1 - err_prob)^(block_size^2)
    # =============================
    recompute_prob = 1 - (1 - err_prob_list) ** (block_size ** 2)

    # =============================
    # 5. ABFTRecompute
    #  ABFTRecompute = Base + (Base - 1) * p/(1-p)
    # =============================
    # ABFTRecompute = ABFTRecomputeBase + (ABFTRecomputeBase - 1) * (recompute_prob / (1 - recompute_prob))
    ABFTRecompute = ABFTRecomputeBase + (ABFTRecomputeBase - 1) * (0.9/V_list)**2 * recompute_prob

    # =============================
    # 6. line_access_ratio
    # =============================
    line_access_ratio = []
    for p in err_prob_list:
        abft_rows, x = get_abft_expected_rows(p, block_size)
        line_access_ratio.append(abft_rows / x)

    line_access_ratio = np.array(line_access_ratio)

    # =============================
    # 7. Ours = (OursBase - 1) + line_access_ratio
    # =============================
    Ours = (OursBase) + 1 * line_access_ratio

    # =============================
    # 8. DMR = DMRBase 
    # =============================
    DMR = DMRBase + k * 1 * recompute_prob

    # =============================
    # 9. 组装 DataFrame
    # =============================
    df = pd.DataFrame({
        "V": V_list,
        "err_prob": err_prob_list,
        "DMRBase": DMRBase,
        "OursBase": OursBase,
        "Stat ABFTBase": ABFTRecomputeBase,
        "recompute_prob": recompute_prob,
        "Stat ABFT": ABFTRecompute,
        "line_access_ratio": line_access_ratio,
        "Ours": Ours,
        "DMR": DMR,
    })

    # 保存
    df.to_excel(output_excel, index=False)
    print(f"生成 Excel 成功: {output_excel}")
    return df


# --------------------
# 示例 main 用法
# --------------------


if __name__ == "__main__":
    # 753G  DiT
    # PixArt 4600G

    # DiT
    # y_scale = 0.669 * 10**9 * 31 * 10**(-12)
    # k = (7.53 * 10**11 * 0.16) / (0.669 * 10**9 * 31) 
    # PixArt
    y_scale = 0.595 * 10**9 * 31 * 10**(-12)
    k = (4.6 * 10**12 * 0.16) / (0.595 * 10**9 * 31)
    # SD1.5
    # y_scale = 1
    # k = 1280 * 0.16 / (y_scale * 31)
    print("compute_change", k, "to", (k * (1.06 *(48/50) * (0.675/0.9)**2 + 2/50)))
    print("energy ratio:", (k * (1.06 *(48/50) * (0.675/0.9)**2 + 2/50) + 1 + 1*0.8) / (k+1)  )
    
    print("weight access power per step:", y_scale)
    print("compute power / weight access power:", k)
    print("compute power per step", k * y_scale)

    excel_path = "test_bar_v_power.xlsx"
    df = generate_barplot_excel(
        vf_curve_path="VFcurve.xlsx",
        output_excel=excel_path,
        k=k,
        block_size=64,
        freq_for_err=1
    )
    print(df)




    plot_grouped_bars_from_excel(
        data_file=excel_path,
        x_col="V",
        plot_columns =["DMR", "Stat ABFT", "Ours"],  # 指定需要画柱的列名
        title="",
        xlabel="Voltage (V)",
        ylabel_left="Energy (J)",
        figsize=(4.2, 2.8),
        dpi=300,
        save_path="ineffectiveness-v-power.svg",
        bar_width=0.7,
        rotation=0,
        y_scale = y_scale * 50
    )
