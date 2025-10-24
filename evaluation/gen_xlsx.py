import os
import json
import pandas as pd
from typing import List, Optional
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

def jsons_to_excel(
    root_folder: str,
    output_excel: str,
    keys_to_write: Optional[List[str]] = None,
    sort_by: Optional[List[str]] = None
):
    """
    遍历 root_folder 的一级子文件夹，读取每个子文件夹下的 run_params.json 并写入 Excel。

    参数：
        root_folder: 存放子文件夹的根目录
        output_excel: 输出 Excel 文件路径
        keys_to_write: 可选，只写入指定的键。如果 None，则写入所有键
        sort_by: 可选，按哪些列降序排序
    """
    records = []

    for subfolder in os.listdir(root_folder):
        sub_path = os.path.join(root_folder, subfolder)
        if not os.path.isdir(sub_path):
            continue

        json_path = os.path.join(sub_path, "run_params.json")
        if not os.path.exists(json_path):
            continue

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 如果 keys_to_write 指定了，就挑选指定键；否则写入所有键
        if keys_to_write is not None:
            record = {k: data.get(k, None) for k in keys_to_write}
        else:
            record = data.copy()

        # 加一列子文件夹名
        record["folder_name"] = subfolder
        records.append(record)

    if not records:
        print("⚠️ 没有找到任何 run_params.json 文件")
        return

    df = pd.DataFrame(records)

    # 排序
    if sort_by:
        for col in sort_by:
            if col not in df.columns:
                continue
            df[col] = pd.to_numeric(df[col], errors="ignore")
        df = df.sort_values(by=sort_by, ascending=False, na_position="last")

    # 格式化数字列（仅显示 4 位有效数字）
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].apply(lambda x: f"{x:.4g}" if pd.notna(x) else None)

    # 写入 Excel
    df.to_excel(output_excel, index=False)
    print(f"✅ 已生成 Excel: {output_excel}")



def clean_excel(
    excel_path: str,
    default_values=None,
    output_clean_excel=None
):
    """
    读取 Excel，将空值、'None'、默认值置为 NaN，并保存清洗后的表格。
    """

    df = pd.read_excel(excel_path)

    # ------------------- 统一处理空值 -------------------
    df = df.replace(["None", "none", "", "[]"], np.nan)

    # ------------------- 统一处理默认值 -------------------
    if default_values:
        for key, dv in default_values.items():
            if key not in df.columns:
                continue
            def is_default(x, dv):
                if pd.isna(x):
                    return True
                # 如果 dv 是 list 或 ndarray
                if isinstance(dv, (list, np.ndarray)):
                    try:
                        if x == dv or str(x) == str(dv):
                            return True
                    except Exception:
                        pass
                else:
                    # 尝试数值比较
                    try:
                        return float(x) == float(dv)
                    except:
                        # 如果不是数值，则用字符串比较
                        return str(x).strip() == str(dv).strip()
                return False

            df[key] = df[key].apply(lambda x: np.nan if is_default(x, dv) else x)

    # ------------------- 保存清洗后的 Excel -------------------
    if output_clean_excel is None:
        folder, fname = os.path.split(excel_path)
        name, ext = os.path.splitext(fname)
        output_clean_excel = os.path.join(folder, f"{name}_clean{ext}")

    df.to_excel(output_clean_excel, index=False)
    print(f"✅ 清洗后的 Excel 已保存到: {output_clean_excel}")
    return df

def plot_from_excel(
    excel_path: str,
    x_keys,              # 主自变量，比如 ["target", "num_inference_steps", "err_prob"]
    y_keys,              # 因变量，比如 ["clip_score","image_reward","lpips_score"]
    optional_keys=None,  # 可选自变量，比如 ["bit","target_layers"]
    default_values=None, # 默认值字典，例如 {"bit": "-1", "target_layers": []}
    log_x_keys=None,     # 哪些自变量用对数坐标，比如 ["err_prob"]
    vertical_x_keys=None # 哪些自变量竖着写，比如['target']
):
    """
    根据 Excel 结果表生成图像：
        same_folder/plots/<可选自变量>/*.png
        same_folder/plots/main/*.png

    每条折线保证其他自变量相同，不同的值会在 label 中标注，label 用小字体画在图外。
    """

    folder = os.path.dirname(excel_path)
    plot_dir = os.path.join(folder, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    if default_values is None:
        default_values = {}

    # ------------------- 1. 清洗数据 -------------------
    df = clean_excel(excel_path, default_values=default_values)
    clean_path = os.path.join(folder, "cleaned.xlsx")
    df.to_excel(clean_path, index=False)
    print(f"✅ 已保存清洗后的 Excel: {clean_path}")

    # ------------------- 2. 可选变量绘图 -------------------

    if optional_keys:
        for opt_key in optional_keys:
            if opt_key not in df.columns:
                print(f"⚠️ 可选变量 {opt_key} 不在 Excel 中，跳过。")
                continue

            df_opt = df.dropna(subset=[opt_key])
            if df_opt.empty:
                print(f"⚠️ 可选变量 {opt_key} 无有效行，跳过。")
                continue

            save_dir = os.path.join(plot_dir, f"{opt_key}")
            os.makedirs(save_dir, exist_ok=True)
            print(f"📊 绘制可选变量 {opt_key} 图，共 {len(df_opt)} 行。")

            for x_key in x_keys:
                if x_key not in df_opt.columns:
                    continue

                fig, axes = plt.subplots(len(y_keys), 1, figsize=(15, 4*len(y_keys)), sharex=True)
                if len(y_keys) == 1:
                    axes = [axes]

                other_x_keys = [k for k in x_keys if k != x_key]
                group_keys = [opt_key] + other_x_keys
                df_opt_grouped = df_opt.dropna(subset=[x_key]+y_keys)
                grouped = df_opt_grouped.groupby(group_keys, dropna=False)

                # 找出公共部分
                all_group_values = {k: set() for k in group_keys}
                for group_vals, _ in grouped:
                    if not isinstance(group_vals, tuple):
                        group_vals = (group_vals,)
                    for k,v in zip(group_keys, group_vals):
                        all_group_values[k].add(v)
                common_keys = [k for k,vset in all_group_values.items() if len(vset)==1]

                colors = matplotlib.colormaps.get_cmap("tab20")(np.linspace(0, 1, len(grouped)))  # tab20 颜色表
                for idx, (group_vals, sub_df) in enumerate(grouped):
                    sub_df = sub_df.sort_values(by=x_key)
                    color = colors[idx]
                    if not isinstance(group_vals, tuple):
                        group_vals = (group_vals,)

                    label_items = [f"{k}={v}" for k,v in zip(group_keys, group_vals) if k not in common_keys]
                    label = ", ".join(label_items) if label_items else None

                    for j, y_key in enumerate(y_keys):
                        if y_key not in sub_df.columns:
                            continue
                        axes[j].plot(sub_df[x_key], sub_df[y_key], "o-",markersize=3.5,  linewidth=1, color=color, label=label)

                for j, y_key in enumerate(y_keys):
                    axes[j].set_ylabel(y_key)
                    axes[j].grid(True)

                axes[-1].set_xlabel(x_key)
                if log_x_keys and x_key in log_x_keys:
                    axes[-1].set_xscale("log")
                if vertical_x_keys and x_key in vertical_x_keys:
                    plt.setp(axes[-1].get_xticklabels(), rotation=90, ha='center')

                handles, labels = axes[0].get_legend_handles_labels()
                if handles:
                    axes[0].legend(handles, labels, fontsize=6, bbox_to_anchor=(1.0, 1), loc='upper left')

                fig.suptitle(f"{opt_key} variation vs {x_key}")
                plt.tight_layout(rect=[0, 0.1, 0.8, 0.97])
                plt.savefig(os.path.join(save_dir, f"{opt_key}_vs_{x_key}.png"))
                plt.close(fig)

            # 去掉已绘制的行
            df = df[df[opt_key].isna()].reset_index(drop=True)

    # ------------------- 3. 主变量绘图 -------------------
    save_dir = os.path.join(plot_dir, "main")
    os.makedirs(save_dir, exist_ok=True)
    print(f"📈 绘制主变量分析图，共 {len(df)} 行。")

    for x_key in x_keys:
        if x_key not in df.columns:
            continue

        fig, axes = plt.subplots(len(y_keys), 1, figsize=(12, 4*len(y_keys)), sharex=True)
        if len(y_keys) == 1:
            axes = [axes]

        other_x_keys = [k for k in x_keys if k != x_key]
        group_keys = other_x_keys
        df_grouped = df.dropna(subset=[x_key]+y_keys)
        if group_keys:
            grouped = df_grouped.groupby(group_keys, dropna=False)
        else:
            grouped = [((), df_grouped)]

        # 找出公共部分
        if group_keys:
            all_group_values = {k: set() for k in group_keys}
            for group_vals, _ in grouped:
                if not isinstance(group_vals, tuple):
                    group_vals = (group_vals,)
                for k,v in zip(group_keys, group_vals):
                    all_group_values[k].add(v)
            common_keys = [k for k,vset in all_group_values.items() if len(vset)==1]
        else:
            common_keys = []
        colors = matplotlib.colormaps.get_cmap("tab20")(np.linspace(0, 1, len(grouped)))
        for idx, (group_vals, sub_df) in enumerate(grouped):
            sub_df = sub_df.sort_values(by=x_key)
            color = colors[idx]
            if group_keys:
                if not isinstance(group_vals, tuple):
                    group_vals = (group_vals,)
                label_items = [f"{k}={v}" for k,v in zip(group_keys, group_vals) if k not in common_keys]
                label = ", ".join(label_items) if label_items else None
            else:
                label = None

            for j, y_key in enumerate(y_keys):
                if y_key not in sub_df.columns:
                    continue
                axes[j].plot(sub_df[x_key], sub_df[y_key], "o-", markersize=3.5,  linewidth=1, color=color, label=label)

        for j, y_key in enumerate(y_keys):
            axes[j].set_ylabel(y_key)
            axes[j].grid(True)

        axes[-1].set_xlabel(x_key)
        if log_x_keys and x_key in log_x_keys:
            axes[-1].set_xscale("log")
        if vertical_x_keys and x_key in vertical_x_keys:
            plt.setp(axes[-1].get_xticklabels(), rotation=90, ha='center')

        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            axes[0].legend(handles, labels, fontsize=6, bbox_to_anchor=(1.0, 1), loc='upper left')

        fig.suptitle(f"Main variable vs {x_key}")
        plt.tight_layout(rect=[0,0.1,0.8,0.97])
        plt.savefig(os.path.join(save_dir, f"main_vs_{x_key}.png"))
        plt.close(fig)

    print(f"✅ 所有图已生成到 {plot_dir}")


if __name__ == "__main__":
    root_folder = "TaylorSeers-Diffusers/taylorseer_flux/results_layers2"
    jsons_to_excel(
        root_folder=root_folder,
        output_excel=os.path.join(root_folder, "summary.xlsx"),
        keys_to_write=None,
        sort_by=["target", "num_inference_steps", "err_prob"]
    )
    plot_from_excel(
        excel_path="TaylorSeers-Diffusers/taylorseer_flux/results_layers2/summary.xlsx",
        x_keys=["target", "num_inference_steps", "err_prob"],
        y_keys=["clip_score", "image_reward_score", "lpips_score"],
        optional_keys=["bit", "target_layers"],
        default_values={"bit": "-1", "target_layers": []},
        log_x_keys=["err_prob"],
        vertical_x_keys=["target"]
    )
