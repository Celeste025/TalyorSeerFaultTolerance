import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from VF_plot import read_vdd_f_excel  # 你已有
from VF_plot import get_ber_value     # 你已有

# -------------------- 工具函数 --------------------
def blend_color(c1, c2, alpha):
    """
    c1, c2: RGB tuple，范围 0~1
    alpha: c2透明度
    返回混合后的 RGB
    """
    return tuple((1 - alpha) * np.array(c1) + alpha * np.array(c2))


def plot_grouped_bars_from_excel(
    data_file,
    x_col,
    plot_columns=None,
    title="",
    xlabel="",
    ylabel_left="",
    figsize=(5, 3.2),
    dpi=500,
    save_path=None,
    transparent=True,
    bar_width=0.3,
    rotation=0,
    extra_alpha=0.6,
    y_scale=1.0
):
    """绘制 Base + Extra 分段柱状图"""
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

    df = pd.read_excel(data_file) if Path(data_file).suffix.lower() in ['.xls', '.xlsx'] else pd.read_csv(data_file)
    if x_col not in df.columns:
        raise ValueError(f"{x_col} not found in {data_file}")

    all_cols = list(df.columns)
    if plot_columns is not None:
        method_cols = [c for c in plot_columns if c in df.columns]
    else:
        method_cols = [c for c in all_cols if c != x_col and not c.endswith("Base")]

    base_cols = {c[:-4]: c for c in all_cols if c.endswith("Base")}

    # scale
    for col in method_cols:
        df[col] = df[col].astype(float) * y_scale
    for base_key, base_col in base_cols.items():
        df[base_col] = df[base_col].astype(float) * y_scale

    x = np.arange(len(df))
    n_methods = len(method_cols)
    total_bar_width = bar_width
    single_bar_width = total_bar_width / n_methods
    offsets = (np.arange(n_methods) - (n_methods - 1) / 2.0) * single_bar_width

    cmap = plt.cm.Set2
    base_colors = cmap([0, 0.5, 0.25])

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    if transparent:
        fig.patch.set_alpha(0.0)
        ax.patch.set_alpha(0.0)

    extra_legend_added = False

    for i, method in enumerate(method_cols):
        vals = df[method].values.astype(float)
        if method in base_cols:
            base_vals = df[base_cols[method]].values.astype(float)
            extra_vals = np.clip(vals - base_vals, 0.0, None)
        else:
            base_vals = vals.copy()
            extra_vals = np.zeros_like(vals)

        # Base
        ax.bar(x + offsets[i], base_vals, width=single_bar_width * 0.93, color=base_colors[i], edgecolor='none', label=method)

        # Extra
        if np.any(extra_vals > 0):
            gray = (0.5, 0.5, 0.5, 1.0)
            blended_color = blend_color(base_colors[i], gray, extra_alpha)
            ax.bar(x + offsets[i], extra_vals, width=single_bar_width * 0.93,
                   bottom=base_vals, color=blended_color, edgecolor='gray', linewidth=0)
            extra_legend_added = True

    y_max = 1.57 * df[method_cols].max().max()
    ax.set_ylim(0, y_max)

    # Legend
    handles, labels = ax.get_legend_handles_labels()
    if extra_legend_added:
        extra_patch = Patch(facecolor='none', edgecolor='gray', label='Recovery Overhead')
        handles.append(extra_patch)
        labels.append('Recovery Overhead')
    ax.legend(handles, labels, frameon=True, fancybox=True, framealpha=0.4,
              edgecolor='gray', loc='upper left', labelspacing=0.15).get_frame().set_linewidth(1)

    ax.set_xticks(x)
    ax.set_xticklabels((df[x_col]*2).astype(str).values, rotation=rotation)
    ax.set_xlabel(xlabel if xlabel else x_col)
    ax.set_ylabel(ylabel_left if ylabel_left else "Value")
    ax.set_title(title, pad=7)
    ax.grid(True, alpha=0.25, linewidth=0.8)
    for spine in ax.spines.values():
        spine.set_linewidth(1)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight',
                    transparent=transparent, facecolor='none' if transparent else 'white')
        print(f"Saved grouped bar plot to: {save_path}")
    plt.close(fig)


# -------------------- ABFT 辅助 --------------------
def get_abft_expected_rows(err_prob, block_size):
    if block_size == 32:
        x, y = 16, 64
    elif block_size == 64:
        x, y = 64, 64
    else:
        raise ValueError("Unsupported block size.")
    p_row_clean = (1 - err_prob) ** y
    p_row_dirty = 1 - p_row_clean
    expected_rows = p_row_dirty * x
    n = x * y
    expected_rows_one = (1 - err_prob) ** (n - 1) * err_prob * n
    abft_expected_rows = expected_rows - expected_rows_one
    return abft_expected_rows, x


# -------------------- 生成 f-power Excel --------------------
def generate_fpower_excel(vf_curve_path="VFcurve.xlsx", output_excel="fpower_data.xlsx",
                          k=7, block_size=64, V_fixed=0.85):
    f_list = np.array([1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8])
    err_prob_list = np.array([get_ber_value(v=V_fixed, f=f) for f in f_list])

    # Base
    DMRBase = 1 + k * (V_fixed / 0.9) ** 2 * 2
    OursBase = 1.1 + k * (V_fixed / 0.9) ** 2 * (1 + 1/32)
    ABFTRecomputeBase = 1 + k * (V_fixed / 0.9) ** 2 * (1 + 1/32)

    # recompute prob
    recompute_prob = 1 - (1 - err_prob_list) ** (block_size ** 2)

    # ABFTRecompute
    ABFTRecompute = ABFTRecomputeBase + (ABFTRecomputeBase - 1) * (0.9 / V_fixed) ** 2 * recompute_prob

    # line access ratio
    line_access_ratio = []
    for p in err_prob_list:
        abft_rows, x = get_abft_expected_rows(p, block_size)
        line_access_ratio.append(abft_rows / x)
    line_access_ratio = np.array(line_access_ratio)

    # Ours
    Ours = OursBase + line_access_ratio

    # DMR
    DMR = DMRBase + k * recompute_prob

    df = pd.DataFrame({
        "f": f_list,
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

    df.to_excel(output_excel, index=False)
    print(f"生成 Excel 成功: {output_excel}")
    return df


# -------------------- main --------------------
if __name__ == "__main__":
    y_scale = 0.669 * 10**9 * 31 * 10**(-12)
    print("weight access power per step:", y_scale)
    k = (7.53 * 10**11 * 0.16) / (0.669 * 10**9 * 31)
    print("compute power / weight access power:", k)
    print("compute power per step:", k * y_scale)

    excel_path = "fpower_data.xlsx"
    df = generate_fpower_excel(vf_curve_path="VFcurve.xlsx", output_excel=excel_path, k=k, block_size=64, V_fixed=0.85)
    print(df)

    plot_grouped_bars_from_excel(
        data_file=excel_path,
        x_col="f",
        plot_columns=["DMR", "Stat ABFT", "Ours"],
        title="",
        xlabel="Frequency (GHz)",
        ylabel_left="Energy (J)",
        figsize=(4.2, 2.8),
        dpi=300,
        save_path="ineffectiveness-f-power.svg",
        bar_width=0.7,
        rotation=0,
        y_scale=y_scale * 50
    )
