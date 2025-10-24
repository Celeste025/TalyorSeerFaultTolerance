import os
import torch
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import pickle
class DiffusionAnalyzer:
    def __init__(
        self,
        save_dir="samples",
        num_channels=4,
        num_pixel_lines=100,
        grid_size=64,
        # ---------- 绘图参数 ----------
        vmin_latent=-2.5, vmax_latent=2.5,
        vmin_pred=-2.5, vmax_pred=2.5,
        vmin_diff=-0.5, vmax_diff=0.5,
        vmin_change=-0.5, vmax_change=0.5,
        vmin_hist=-0.3, vmax_hist=0.3,
        # ---------- 绘图开关 ----------
        plot_heatmap=True,
        plot_diff=True,
        plot_hist=True,
        plot_lines=True,
        plot_alpha = True,
        save_data = True
    ):
        """
        数据分析和作图类
        """
        self.save_dir = save_dir
        self.num_channels = num_channels
        self.num_pixel_lines = num_pixel_lines
        self.grid_size = grid_size

        # ---------- 绘图参数 ----------
        self.vmin_latent = vmin_latent
        self.vmax_latent = vmax_latent
        self.vmin_pred = vmin_pred
        self.vmax_pred = vmax_pred
        self.vmin_diff = vmin_diff
        self.vmax_diff = vmax_diff
        self.vmin_change = vmin_change
        self.vmax_change = vmax_change
        self.vmin_hist = vmin_hist
        self.vmax_hist = vmax_hist

        # ---------- 绘图开关 ----------
        self.plot_heatmap = plot_heatmap
        self.plot_diff = plot_diff
        self.plot_hist = plot_hist
        self.plot_lines = plot_lines
        self.plot_alpha = plot_alpha
        # ---------- 精确数据保存 ----------
        self.save_data = save_data
        # 创建保存目录
        self.precise_data = {
            'steps': [],
            'latents': [],
            'pred_xstarts': [],
            'alpha_bars': [],
            'noises': []  # 如果需要记录噪声，可以在这里添加
        }
        # ---------- 创建文件夹 ----------
        self.folders = {
            "latent": os.path.join(save_dir, "latent"),
            "fig": os.path.join(save_dir, "fig"),
            "pred": os.path.join(save_dir, "pred"),
            "noise": os.path.join(save_dir, "noise"),
            "pred_xstart_latent": os.path.join(save_dir, "pred_xstart-latent"),
            "pred_change": os.path.join(save_dir, "pred_change"),
            "noise_change": os.path.join(save_dir, "noise_change"),
            "latent_change": os.path.join(save_dir, "latent_change"),
            "latent_lines": os.path.join(save_dir, "latent_lines"),
            "pred_lines": os.path.join(save_dir, "pred_lines"),
            "hist_pred": os.path.join(save_dir, "hist_pred"),
            "hist_latent": os.path.join(save_dir, "hist_latent"),
            "pred-latent_change": os.path.join(save_dir, "pred-latent_change"),
            "noise_lines": os.path.join(save_dir, "noise_lines")
        }
        for f in self.folders.values():
            os.makedirs(f, exist_ok=True)

        # ---------- 固定像素点 ----------
        self.pixel_positions = self._init_pixel_positions()

        # ---------- 历史记录 ----------
        self.latent_history = [[] for _ in range(num_channels * num_pixel_lines)]
        self.pred_history = [[] for _ in range(num_channels * num_pixel_lines)]
        self.noise_history = [[] for _ in range(num_channels * num_pixel_lines)]
        
        self.alpha_bar_history = []

        self.prev_latents = None
        self.prev_pred_xstart = None

    # --------------------------
    # 初始化像素点
    # --------------------------
    def _init_pixel_positions(self):
        pixel_positions = []
        step = self.grid_size // int(np.sqrt(self.num_pixel_lines))
        for ch in range(self.num_channels):
            pos = []
            for i in range(0, self.grid_size, step):
                for j in range(0, self.grid_size, step):
                    if len(pos) < self.num_pixel_lines:
                        pos.append((i, j))
            pixel_positions.append(pos)
        return pixel_positions


    def _save_precise_step_data(self, latents, pred_xstart, step_idx, alpha_bar_t, noise):
        """保存每一步的精确数据"""
        step_data = {
            'step_idx': step_idx,
            'latent': latents.detach().cpu().clone(),  # 使用clone确保数据独立
            'pred_xstart': pred_xstart.detach().cpu().clone(),
            'alpha_bar': alpha_bar_t,
            'noise': noise.detach().cpu().clone() if noise is not None else None,
            'timestamp': torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        }
        
        # 添加到总数据中
        self.precise_data['steps'].append(step_idx)
        self.precise_data['latents'].append(step_data['latent'])
        self.precise_data['pred_xstarts'].append(step_data['pred_xstart'])
        self.precise_data['alpha_bars'].append(alpha_bar_t)
        self.precise_data['noises'].append(step_data['noise'])

    # --------------------------
    # 记录每步数据并绘图
    # --------------------------
    def record_step(self, latents, pred_xstart, step_idx, vae=None, alpha_bar_t=None, noise=None):
        #在有batch的情况下也只记录每个batch的第一张，方便处理，避免保存过多数据
        latents = latents.squeeze()
        if latents.dim() == 2:
            latents = latents.unsqueeze(0)
        pred_xstart = pred_xstart.squeeze()
        if pred_xstart.dim() == 2:
            pred_xstart = pred_xstart.unsqueeze(0)
        if noise is not None:
            noise = noise.squeeze()
            if noise.dim() == 2:
                noise = noise.unsqueeze(0)
        
        if latents.dim() == 4:
            latents = latents[0]
        if pred_xstart.dim() == 4:
            pred_xstart = pred_xstart[0]
        if noise is not None and noise.dim() == 4:
            noise = noise[0]

        # 检查一下维度是否都为3了
        if latents.dim() != 3 or pred_xstart.dim() != 3 or (noise is not None and noise.dim() != 3):
            raise ValueError(f"Latents, pred_xstart, and noise must be 3D tensors after squeezing. Got: latents {latents.shape}, pred_xstart {pred_xstart.shape}, noise {noise.shape if noise is not None else 'N/A'}")


        latents_cpu = latents[:self.num_channels].cpu().float().numpy()
        pred_cpu = pred_xstart[:self.num_channels].cpu().float().numpy()
        noise_cpu = noise[:self.num_channels].cpu().float().numpy()
        if self.save_data:
            self._save_precise_step_data(latents, pred_xstart, step_idx, alpha_bar_t, noise)

        # ---------- 绘制热力图 ----------
        if self.plot_heatmap:
            self._plot_latent(latents_cpu, step_idx)
            self._plot_pred(pred_cpu, step_idx)
            self._plot_noise(noise_cpu, step_idx) 

        # ---------- 绘制差值热力图 ----------
        if self.plot_diff:
            self._plot_pred_xstart_minus_latent(latents_cpu, pred_cpu, step_idx)
            if self.prev_latents is not None and self.prev_pred_xstart is not None:
                self._plot_change_latent(latents, step_idx)
                self._plot_change_pred(pred_xstart, step_idx)
                self._plot_change_noise(noise, step_idx)
                self._plot_change_pred_minus_latent(pred_xstart, latents, step_idx)

        # ---------- 绘制直方图 ----------
        if self.plot_hist and self.prev_latents is not None and self.prev_pred_xstart is not None:
            self._plot_hist_pred(pred_xstart, step_idx)
            self._plot_hist_latent(latents, step_idx)

        # ---------- 保存 VAE 解码图像 ----------
        if vae is not None:
            self._save_vae_image(latents, vae, step_idx)

        # ---------- 保存折线数据 ----------
        self._record_pixel_lines(latents, pred_xstart)

        # ---------- 更新上一轮 ----------
        self.prev_latents = latents.clone()
        self.prev_pred_xstart = pred_xstart.clone()

        if alpha_bar_t is not None:
            self.alpha_bar_history.append(alpha_bar_t)

    # ======================
    # 内部绘图函数
    # ======================
    # 热力图函数
    def _plot_latent(self, data, step_idx):
        self._plot_channels(data, step_idx, "latent", self.vmin_latent, self.vmax_latent)
    def _plot_noise(self, data, step_idx):
        self._plot_channels(data, step_idx, "noise", self.vmin_latent, self.vmax_latent)
    def _plot_pred(self, data, step_idx):
        self._plot_channels(data, step_idx, "pred", self.vmin_pred, self.vmax_pred)

    def _plot_pred_xstart_minus_latent(self, latents, pred, step_idx):
        diff = pred - latents
        self._plot_channels(diff, step_idx, "pred_xstart_latent", self.vmin_diff, self.vmax_diff)

    def _plot_change_latent(self, latents, step_idx):
        diff = (latents - self.prev_latents)[:self.num_channels].cpu().float().numpy()
        self._plot_channels(diff, step_idx, "latent_change", self.vmin_change, self.vmax_change)

    def _plot_change_pred(self, pred_xstart, step_idx):
        diff = (pred_xstart - self.prev_pred_xstart)[:self.num_channels].cpu().float().numpy()
        self._plot_channels(diff, step_idx, "pred_change", self.vmin_change, self.vmax_change)
    
    def _plot_change_noise(self, noise, step_idx):
        diff = (noise - self.prev_latents)[:self.num_channels].cpu().float().numpy()
        self._plot_channels(diff, step_idx, "noise_change", self.vmin_change, self.vmax_change)
    
    def _plot_change_pred_minus_latent(self, pred_xstart, latents, step_idx):
        diff = ((pred_xstart - latents) - (self.prev_pred_xstart - self.prev_latents))[:self.num_channels].cpu().float().numpy()
        self._plot_channels(diff, step_idx, "pred-latent_change", self.vmin_change, self.vmax_change)

    # 绘制折线
    def _record_pixel_lines(self, latents, pred_xstart, noise=None):
        if len(latents.shape) != 3 or len(pred_xstart.shape) != 3:
            raise ValueError(f"Latents and pred_xstart should be 3D tensors with shape (channels, height, width). Yet got shapes {latents.shape} and {pred_xstart.shape}.")
        for ch in range(self.num_channels):
            for idx, (x, y) in enumerate(self.pixel_positions[ch]):
                flat_idx = ch*self.num_pixel_lines + idx
                self.latent_history[flat_idx].append(latents[ch, x, y].item())
                self.pred_history[flat_idx].append(pred_xstart[ch, x, y].item())
                self.noise_history[flat_idx].append(noise[ch, x, y].item() if noise is not None else 0.0)

    # 绘制直方图
    def _plot_hist_pred(self, pred_xstart, step_idx):
        diff_flat = (pred_xstart - self.prev_pred_xstart)[:self.num_channels].cpu().float().numpy().flatten()
        self._plot_hist(diff_flat, step_idx, "hist_pred", color='blue', vmin=self.vmin_hist, vmax=self.vmax_hist)

    def _plot_hist_latent(self, latents, step_idx):
        diff_flat = (latents - self.prev_latents)[:self.num_channels].cpu().float().numpy().flatten()
        self._plot_hist(diff_flat, step_idx, "hist_latent", color='red', vmin=self.vmin_hist, vmax=self.vmax_hist)

    # 通用绘图函数
    def _plot_channels(self, data, step_idx, folder_key, vmin, vmax):
        if len(data.shape) != 3:
            raise ValueError(
                f"Data should be a 3D array with shape (channels, height, width) for func _plot_channels. "
                f"Yet got shape {data.shape}."
            )
        if data.shape[0] != self.num_channels:
            print(f"Warning: Data has shape {data.shape}, expected {self.num_channels}.")
        fig, axes = plt.subplots(2, 2, figsize=(8, 8))
        for ch in range(self.num_channels):
            ax = axes[ch // 2, ch % 2]
            im = ax.imshow(data[ch], cmap="viridis", vmin=vmin, vmax=vmax)
            ax.set_title(f"Channel {ch}")
            ax.axis("off")

            # === 计算 min/max ===
            ch_min = data[ch].min().item()
            ch_max = data[ch].max().item()

            # 在 colorbar 上加标注
            cbar = fig.colorbar(im, ax=ax, shrink=0.6)
            cbar.set_label(f"min={ch_min:.2f}, max={ch_max:.2f}", fontsize=8)

        plt.suptitle(f"Step {step_idx} {folder_key}")
        plt.tight_layout()
        plt.savefig(os.path.join(self.folders[folder_key], f"{folder_key}_{step_idx}.png"))
        plt.close()

    def _plot_hist(self, values, step_idx, folder_key, color, vmin, vmax):
        plt.figure(figsize=(6,4))
        plt.hist(values, bins=50, color=color, alpha=0.7, range=(vmin, vmax))
        plt.title(f"Step {step_idx} {folder_key} Histogram")
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(os.path.join(self.folders[folder_key], f"hist_{step_idx}.png"))
        plt.close()

    def _save_vae_image(self, latents, vae, step_idx):
        if len(latents.shape) == 3:
            latents = latents.unsqueeze(0)
        if len(latents.shape) != 4:
            raise ValueError(f"Latents should be a 4D tensor with shape (batch, channels, height, width) for VAE decoding. Yet got shape {latents.shape}.")
        latents_vae = latents / 0.18215
        image = vae.decode(latents_vae).sample
        image = (image / 2 + 0.5).clamp(0,1)
        image = image.cpu().permute(0,2,3,1).float().numpy()[0]
        image = (image*255).astype("uint8")
        Image.fromarray(image).save(os.path.join(self.folders["fig"], f"fig_{step_idx}.png"))

    # 绘制折线图
    def plot_pixel_lines(self, step_base=True, theta_base=True):
        # 绘制 latent 折线图
        for ch in range(self.num_channels):
            if step_base:
                plt.figure(figsize=(8,6))
                for idx in range(self.num_pixel_lines):
                    flat_idx = ch*self.num_pixel_lines + idx
                    plt.plot(range(len(self.latent_history[flat_idx])), 
                            self.latent_history[flat_idx])
                plt.xlabel("Step")
                plt.ylabel("Value")
                plt.title(f"Latent Channel {ch} Pixel Value Over Steps")
                plt.tight_layout()
                plt.savefig(os.path.join(self.folders["latent_lines"], f"latent_channel_{ch}.png"))
                plt.close()
            
            if theta_base:
                plt.figure(figsize=(8,6))
                for idx in range(self.num_pixel_lines):
                    flat_idx = ch*self.num_pixel_lines + idx
                    theta_base_x = [np.arccos(np.sqrt(1-a)) for a in self.alpha_bar_history]
                    plt.plot(theta_base_x, 
                            self.latent_history[flat_idx])
                plt.xlabel("theta_step")
                plt.ylabel("Value")
                plt.title(f"Latent Channel {ch} Pixel Value Over Theta-steps")
                plt.tight_layout()
                plt.savefig(os.path.join(self.folders["latent_lines"], f"latent_channel_{ch}_theta_base.png"))
                plt.close()

        # 绘制 pred_xstart 折线图
        for ch in range(self.num_channels):
            if step_base:
                plt.figure(figsize=(8,6))
                for idx in range(self.num_pixel_lines):
                    flat_idx = ch*self.num_pixel_lines + idx
                    plt.plot(range(len(self.pred_history[flat_idx])), 
                            self.pred_history[flat_idx])
                plt.xlabel("Step")
                plt.ylabel("Value")
                plt.title(f"Pred_xstart Channel {ch} Pixel Value Over Steps")
                plt.tight_layout()
                plt.savefig(os.path.join(self.folders["pred_lines"], f"pred_channel_{ch}.png"))
                plt.close()
            
            if theta_base:
                plt.figure(figsize=(8,6))
                for idx in range(self.num_pixel_lines):
                    flat_idx = ch*self.num_pixel_lines + idx
                    theta_base_x = [np.arccos(np.sqrt(1-a)) for a in self.alpha_bar_history]
                    plt.plot(theta_base_x, 
                            self.pred_history[flat_idx])
                plt.xlabel("theta_step")
                plt.ylabel("Value")
                plt.title(f"Pred_xstart Channel {ch} Pixel Value Over Theta-steps")
                plt.tight_layout()
                plt.savefig(os.path.join(self.folders["pred_lines"], f"pred_channel_{ch}_theta_base.png"))
                plt.close()
        # 绘制noise折线图
        for ch in range(self.num_channels):
            if step_base:
                plt.figure(figsize=(8,6))
                for idx in range(self.num_pixel_lines):
                    flat_idx = ch*self.num_pixel_lines + idx
                    plt.plot(range(len(self.noise_history[flat_idx])), 
                            self.noise_history[flat_idx])
                plt.xlabel("Step")
                plt.ylabel("Value")
                plt.title(f"Noise Channel {ch} Pixel Value Over Steps")
                plt.tight_layout()
                plt.savefig(os.path.join(self.folders["noise_lines"], f"noise_channel_{ch}.png"))
                plt.close()
            
            if theta_base:
                plt.figure(figsize=(8,6))
                for idx in range(self.num_pixel_lines):
                    flat_idx = ch*self.num_pixel_lines + idx
                    theta_base_x = [np.arccos(np.sqrt(1-a)) for a in self.alpha_bar_history]
                    plt.plot(theta_base_x, 
                            self.noise_history[flat_idx])
                plt.xlabel("theta_step")
                plt.ylabel("Value")
                plt.title(f"Noise Channel {ch} Pixel Value Over Theta-steps")
                plt.tight_layout()
                plt.savefig(os.path.join(self.folders["noise_lines"], f"noise_channel_{ch}_theta_base.png"))
                plt.close()

    # 绘制alpha_bar曲线分析
    def plot_alpha_lines(self):
        if len(self.alpha_bar_history) == 0:
            print("Warning: No alpha_bar data to plot.")
        else:
            plt.figure(figsize=(8,6))
            plt.plot(range(len(self.alpha_bar_history)), self.alpha_bar_history, label="alpha_bar")
            plt.plot(range(len(self.alpha_bar_history)), 
                    [np.sqrt(a) for a in self.alpha_bar_history], label="sqrt(alpha_bar)")
            plt.plot(range(len(self.alpha_bar_history)), 
                    [np.sqrt(1 - a) for a in self.alpha_bar_history], label="sqrt(1 - alpha_bar)")
            plt.xlabel("Step")
            plt.ylabel("Alpha Bar Value")
            plt.title("Alpha Bar Over Steps")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(self.save_dir, "alpha_bar_over_steps.png"))
            plt.close()
            
    def end_plot(self):
        if self.plot_lines:
            self.plot_pixel_lines()
        if self.plot_alpha:
            self.plot_alpha_lines()
        if self.save_data:
            self.save_data_to_disk()

    def save_data_to_disk(self, file_name="precise_data.pkl"):
        with open(os.path.join(self.save_dir, file_name), "wb") as f:
            pickle.dump(self.precise_data, f)
        print(f"Precise data saved to {os.path.join(self.save_dir, file_name)}")

    @staticmethod
    def load_precise_data(filepath):
        """从文件加载精确数据"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        return data


_diffusion_analyzer = DiffusionAnalyzer()