import torch
import os
from typing import Dict

class HookManager:
    def __init__(self):
        """
        Args:
            model: PyTorch模型
            layer_names: 要hook的层名称列表，如果为None则hook所有层
        """
        self.model = None
        self.layer_names = []
        self.records: Dict[str, Dict[str, torch.Tensor]] = {}  # 存储输入输出
        self.hooks = []
    
    def initialize(self, model, layer_names):
        self.model = model
        self.layer_names = layer_names 
    
    def _make_hook(self, name: str, capture_mode: str = "both"):
        """
        capture_mode: 'input', 'output', 'both'
        """
        def hook(module, input, output):
            entry = {}
            if capture_mode in ("input", "both"):
                # input 是 tuple，这里只取第一个，如果想要全部可改成 input_tuple = input
                entry["input"] = tuple(
                    inp.detach().clone().cpu() if isinstance(inp, torch.Tensor) else inp
                    for inp in input
                )
            if capture_mode in ("output", "both"):
                # output 可能是 tuple 或单 tensor
                if isinstance(output, tuple):
                    entry["output"] = tuple(
                        out.detach().clone().cpu() if isinstance(out, torch.Tensor) else out
                        for out in output
                    )
                else:
                    entry["output"] = output.detach().clone().cpu()
            self.records[name] = entry
        return hook

    def register_hooks(self, capture_mode: str = "both", print_module_names: bool = False):
        """
        注册hook
        Args:
            capture_mode: 'input', 'output', 'both'
            print_module_names: 是否打印模型所有层名
        """
        for name, module in self.model.named_modules():
            if print_module_names:
                print(name)
            if self.layer_names is None or name in self.layer_names:
                hook = module.register_forward_hook(self._make_hook(name, capture_mode))
                self.hooks.append(hook)
        return self

    def remove_hooks(self):
        """移除所有hook"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def clean_records(self):
        """清空记录"""
        self.records = {}

    def save_records(self, save_dir: str, save_name: str):
        """保存记录到文件"""
        if not self.records:
            print("Warning: no records to save.")
            return ""
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir, save_name)
        torch.save(self.records, save_path)
        print("Saved layer records to", save_path)
        self.clean_records()
        return save_path

    @staticmethod
    def load_records(file_path: str) -> Dict[str, Dict[str, torch.Tensor]]:
        """从文件加载记录"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"{file_path} does not exist.")
        return torch.load(file_path)

    def __call__(self, x: torch.Tensor) -> Dict[str, Dict[str, torch.Tensor]]:
        """执行前向传播并返回记录"""
        self.clean_records()
        with torch.no_grad():
            _ = self.model(x)
        return self.records


# 单例（与原来相同）
_hook_manager = HookManager()
