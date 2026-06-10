# plot_latent_heatmaps.py
import os
import re
import argparse
from pathlib import Path
import math

import torch
import numpy as np
import matplotlib.pyplot as plt

def find_pt_files(folder):
    p = Path(folder)
    files = [f for f in p.iterdir() if f.is_file() and f.suffix == ".pt"]
    # try to filter step/latent style names first
    pattern = re.compile(r"(?:step|latent)[^\d]*?(\d+)", re.IGNORECASE)
    scored = []
    for f in files:
        m = pattern.search(f.name)
        if m:
            key = int(m.group(1))
        else:
            # fallback: sort by name
            key = float("inf")
        scored.append((key, f))
    # sort by extracted key then by filename
    scored.sort(key=lambda x: (x[0], x[1].name))
    return [f for _, f in scored]

def tensor_to_2d(tensor):
    """
    Collapse tensor to 2D array for heatmap:
    - (B, C, H, W) -> mean over B and C -> (H, W)
    - (C, H, W) -> mean over C -> (H, W)
    - (H, W) -> (H, W)
    - (N,) or (K,) -> reshape to (1, K)
    Returns numpy float32 array.
    """
    t = tensor
    if isinstance(t, torch.Tensor):
        t = t.detach().cpu().float().numpy()
    else:
        t = np.array(t, dtype=np.float32)

    if t.ndim == 4:
        # (B, C, H, W)
        arr = t.mean(axis=(0,1))
    elif t.ndim == 3:
        # (C, H, W)
        arr = t.mean(axis=0)
    elif t.ndim == 2:
        arr = t
    elif t.ndim == 1:
        arr = t.reshape(1, -1)
    else:
        # collapse all leading dims except last two if possible
        # fallback: mean over all but last two dims
        while arr.ndim > 2:
            t = t.mean(axis=0)
            if isinstance(t, np.ndarray):
                arr = t
            else:
                arr = np.array(t)
        arr = arr
    # ensure 2D
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr

def plot_and_save_heatmap(arr2d, outpath, cmap="viridis", vmin=None, vmax=None, dpi=300):
    plt.figure(figsize=(arr2d.shape[1]/100 + 1.2, arr2d.shape[0]/100 + 1.2), dpi=dpi)
    ax = plt.gca()
    im = ax.imshow(arr2d, aspect='auto', origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(im, fraction=0.046, pad=0.01)
    plt.tight_layout()
    plt.savefig(outpath, dpi=dpi, bbox_inches='tight')
    plt.close()

def make_grid(all_arrs, outpath, cols=6, cmap="viridis", vmin=None, vmax=None, dpi=300):
    n = len(all_arrs)
    if n == 0:
        return
    cols = max(1, cols)
    rows = math.ceil(n / cols)
    # choose common cell size
    cell_w = 2.0
    cell_h = 2.0
    fig, axes = plt.subplots(rows, cols, figsize=(cols*cell_w, rows*cell_h), dpi=dpi)
    axes = np.array(axes).reshape(-1)
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    for i, arr in enumerate(all_arrs):
        ax = axes[i]
        im = ax.imshow(arr, aspect='auto', origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(f"step {i}", fontsize=8)
    # hide any unused axes
    for j in range(len(all_arrs), len(axes)):
        axes[j].axis('off')
    # colorbar on the right using last axis
    fig.subplots_adjust(right=0.92, top=0.95, bottom=0.05)
    cax = fig.add_axes([0.94, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cax)
    plt.savefig(outpath, dpi=dpi, bbox_inches='tight')
    plt.close()

def main(args):
    files = find_pt_files(args.input_dir)
    if not files:
        print("No .pt files found in", args.input_dir)
        return

    os.makedirs(args.out_dir, exist_ok=True)
    all_arrs = []
    vmin, vmax = args.vmin, args.vmax
    # optional two-pass to compute global vmin/vmax if requested 'auto' and grid mode
    if args.global_scale:
        vals = []
        for f in files:
            try:
                t = torch.load(f, map_location="cpu")
                arr = tensor_to_2d(t)
                vals.append(arr)
            except Exception as e:
                print("Failed to load", f, ":", e)
        if vals:
            stacked = np.concatenate([a.flatten() for a in vals])
            vmin = float(np.nanpercentile(stacked, args.global_vmin_percent)) if args.global_vmin_percent is not None else float(np.nanmin(stacked))
            vmax = float(np.nanpercentile(stacked, args.global_vmax_percent)) if args.global_vmax_percent is not None else float(np.nanmax(stacked))
            print(f"Global vmin/vmax set to {vmin:.6g} / {vmax:.6g}")

    for idx, f in enumerate(files):
        fname = f.name
        try:
            t = torch.load(f, map_location="cpu")
        except Exception as e:
            print("Failed to torch.load", f, ":", e)
            continue
        arr2d = tensor_to_2d(t)
        all_arrs.append(arr2d)
        # save single heatmap
        out_single = os.path.join(args.out_dir, f"{Path(fname).stem}_heatmap.png")
        plot_and_save_heatmap(arr2d, out_single, cmap=args.cmap, vmin=vmin, vmax=vmax, dpi=args.dpi)
        print("Saved heatmap:", out_single)

    # if requested, make a grid of all heatmaps (useful for small number of steps)
    if args.make_grid:
        grid_path = os.path.join(args.out_dir, "all_heatmaps.png")
        make_grid(all_arrs, grid_path, cols=args.cols, cmap=args.cmap, vmin=vmin, vmax=vmax, dpi=args.dpi)
        print("Saved grid heatmap:", grid_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot heatmaps from saved .pt latent tensors (step files).")
    parser.add_argument("--input_dir", type=str, default="/data/home/jinqiwen/workspace/diffusion_fault_tolerance/TaylorSeerFaultTolerance/PixArt/results/target_Skip_step_30_err_prob_0.0_h/images_gen/layer_out",help="Folder containing .pt files (step_*.pt etc).")
    parser.add_argument("--out_dir", type=str, default="./heatmaps", help="Where to save heatmap pngs.")
    parser.add_argument("--cmap", type=str, default="viridis", help="Matplotlib colormap for heatmaps.")
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--make_grid", action="store_true", help="Also create a grid image of all heatmaps.")
    parser.add_argument("--cols", type=int, default=6, help="Columns in the grid image.")
    parser.add_argument("--global_scale", action="store_true", help="Compute global vmin/vmax across files for consistent color scale.")
    parser.add_argument("--global_vmin_percent", type=float, default=1.0, help="Percentile for global vmin if global_scale set (default 1%%).")
    parser.add_argument("--global_vmax_percent", type=float, default=99.0, help="Percentile for global vmax if global_scale set (default 99%%).")
    parser.add_argument("--vmin", type=float, default=None, help="Manual vmin for colormap.")
    parser.add_argument("--vmax", type=float, default=None, help="Manual vmax for colormap.")
    args = parser.parse_args()

    # If user provided explicit vmin/vmax via args, override
    if args.vmin is not None:
        args.vmin = args.vmin
    if args.vmax is not None:
        args.vmax = args.vmax

    main(args)