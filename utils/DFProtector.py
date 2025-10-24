import os
import math
import numpy as np
import torch 
import enum

# 用于diffusion step之间信息提取和错误保护，不涉及噪声预测模型内部。
# 纠错需要存储diffusion step的结束后输出的1.sample 2.pred_xstart
class DFProtector:
    def __init__(self):
        self.thre = dict()
        # thre keys will be like "sample", "pred_xstart" -> (min_val, max_val)
        self.buffer = dict()
        self.buffer["sample"] = []  # 记录每个step的sample (numpy arrays)
        self.buffer["pred_xstart"] = []  # 记录每个step的pred_xstart (numpy arrays)
        self.slack = 1.3  # 阈值放宽比例
        self.correct_mode = "no"  # "zero", "prev", "k", "no"
        self.prev_thre = (-0.4, 0.4)
        self.k_thre = (-0.1, 0.1)
    
    def set_mode(self, mode):
        if mode not in ["zero", "prev", "k", "no"]:
            raise ValueError(f"Unknown correction mode '{mode}'")
        self.correct_mode = mode
        print(f"Set correction mode to '{mode}'")

    
    def record(self, data):
        for key, value in data.items():
            if key not in self.buffer:
                raise ValueError(f"Conveying data key '{key}' not recognized.")
            if isinstance(value, torch.Tensor):
                self.buffer[key].append(value.detach().cpu().numpy())
            else:
                self.buffer[key].append(value)

    def compute_thresholds(self):
        """
        高效计算每个 buffer 的全局 min/max（CPU 上）
        - 支持 numpy.ndarray 和 torch.Tensor
        - 使用 torch.stack 保留原始维度
        - 避免 Python 循环逐元素计算
        """
        for buf_name, hist in self.buffer.items():
            if not hist:
                print(f"[compute_thresholds] buffer '{buf_name}' is empty, skipping.")
                continue

            # 将所有条目统一转成 torch.Tensor
            tensors = []
            for arr in hist:
                if isinstance(arr, torch.Tensor):
                    tensors.append(arr)
                else:  # numpy 或其他可转换类型
                    tensors.append(torch.tensor(arr))

            # 如果所有条目形状相同，直接 stack 求全局 min/max
            try:
                all_tensor = torch.stack(tensors)  # shape: (num_entries, ...)
            except RuntimeError:
                # 条目形状不一致，只能 flatten 再 cat
                all_tensor = torch.cat([t.reshape(-1) for t in tensors], dim=0)

            # 计算全局 min/max
            min_val = float(all_tensor.min().item())
            max_val = float(all_tensor.max().item())

            self.thre[buf_name] = (min_val, max_val)
            print(f"[compute_thresholds] buffer '{buf_name}': min={min_val:.6f}, max={max_val:.6f}")

    # -------------------------
    # _get_mask: return boolean mask of positions in latest record that are outside thresholds
    # -------------------------
    def _get_mask(self, buffer_name, step):  #step  eg (0,30)
        """
        Return a boolean mask (numpy array) for the latest entry in buffer[buffer_name],
        True means value outside stored thresholds -> anomalous.
        """
        if buffer_name not in self.buffer or len(self.buffer[buffer_name]) == 0:
            raise ValueError(f"Buffer '{buffer_name}' empty or not exist.")

        xt_0 = self.buffer[buffer_name][-1]  # latest
        xt_1 = self.buffer[buffer_name][-2] if len(self.buffer[buffer_name]) >= 2 else None
        xt_2 = self.buffer[buffer_name][-3] if len(self.buffer[buffer_name]) >= 3 else None

        if buffer_name not in self.thre.keys():
            raise KeyError(f"No thresholds computed for '{buffer_name}'. Call compute_thresholds() first.")

        min_val, max_val = self.thre[buffer_name]  # 标量

        # 1. xt_0 超出最大最小值
        mask0 = (xt_0 < (min_val * self.slack)) | (xt_0 > (max_val * self.slack))

        mask1 = np.zeros_like(mask0, dtype=bool)  # 全部为0
        if xt_1 is not None:
            mask1 = (xt_0 - xt_1 < -0.07) | (xt_0 - xt_1 > 0.07)

        # 3. 二阶差值异常: 2*xt_1 - xt_0 - xt_2 超出 self.k_thre
        mask2 = np.zeros_like(mask0, dtype=bool)
        scale = max(1 + 1.5 * (step[0] / step[1]), 1)  # 当前step/总step, 越后面scale越大
        k_thre = (self.k_thre[0]*scale, self.k_thre[1]*scale)
        if xt_1 is not None and xt_2 is not None:
            mask2 = (2*xt_1 - xt_0 - xt_2 < k_thre[0]) | (2*xt_1 - xt_0 - xt_2 > k_thre[1])

        # 最终 mask: mask0 单独判定，mask1 和 mask2 同时为 True 才判定异常
        mask = mask0 | mask2
        if step[0] == step[1] - 1:
            mask = mask | mask1  # 最后一步也考虑 mask1
        return mask

    # -------------------------
    #  correct: correct the latest record according to mask and chosen mode
    #    modes: "zero", "prev"
    # -------------------------
    def correct(self, step, buffer_name="sample" , quiet=True):   #step(0,30)总扩散步骤为30，当前是第0step
        """
        Correct anomalous positions in the latest buffer entry according to thresholds.
        The latest entry in self.buffer[buffer_name] will be replaced by the corrected version.
        Args:
            buffer_name: which buffer to operate on (e.g., "sample")
            mode: "zero" | "prev" 
                  - "zero": set anomalous positions to 0
                  - "prev": set anomalous positions to previous time-step values 
                  - "k" : set to 2*xt-1 - xt-2  
        Returns:
            corrected_data: corrected latest record (numpy now)
        """
        if buffer_name not in self.buffer or len(self.buffer[buffer_name]) == 0:
            raise ValueError(f"Buffer '{buffer_name}' empty or not exist.")

        latest = self.buffer[buffer_name][-1].copy()  # numpy array
        mask = self._get_mask(buffer_name=buffer_name, step=step) # boolean numpy array
        if not quiet:
            num_anom = np.sum(mask)
            total = mask.size
            print(f"[correct] Buffer '{buffer_name}': {num_anom}/{total} ({num_anom/total*100:.2f}%) anomalous positions detected.")
        # 检查一下mask和latest形状是否匹配
        if mask.shape != latest.shape:
            raise ValueError(f"Mask shape {mask.shape} does not match latest data shape {latest.shape}.")
        mode = self.correct_mode
        if step[0] == step[1] - 1:
            mode = "prev"  #最后一步强行置为prev mode

        if mode == "zero":
            latest[mask] = 0
        elif mode == "prev":
            if len(self.buffer[buffer_name]) >= 2:
                prev = self.buffer[buffer_name][-2]
                latest[mask] = prev[mask]
            else:
                latest[mask] = 0       
        elif mode == "k":
            if len(self.buffer[buffer_name]) >= 3:
                prev1 = self.buffer[buffer_name][-2]
                prev2 = self.buffer[buffer_name][-3]
                # print(latest.shape, prev1.shape, prev2.shape, mask.shape)
                latest[mask] = (2 * prev1 - prev2)[mask]
            elif len(self.buffer[buffer_name]) == 2:
                prev = self.buffer[buffer_name][-2]
                latest[mask] = prev[mask]
            else:
                latest[mask] = 0
            latest = np.clip(latest, 1.2 * self.thre[buffer_name][0], 1.2 * self.thre[buffer_name][1])  # 防止纠正后数值过大

        elif mode == "no":
            pass
        else:
            raise ValueError(f"Unknown mode '{mode}'")

        
        # replace the latest entry with corrected result
        self.buffer[buffer_name][-1] = latest
        return self.buffer[buffer_name][-1]

    # -------------------------
    # 清空记录的数据, 每个diffusion过程开始前调用
    # -------------------------
    def clear(self):
        for key in self.buffer.keys():
            self.buffer[key] = []
        # do not clear thresholds by default; keep them unless explicitly needed
    
    def deep_clear(self):
        self.clear()
        self.thre = dict()


_df_protector = DFProtector()