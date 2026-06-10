import os 
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR))

from fake_quant import NoisyW8A8Conv2d, W8A8Conv2d, W8A8Linear, W8A8BMM, NoisyW8A8Linear, NoisyW8A8BMM, NoisyW8A8BMM, NoisyW8A8LinearProtected, NoisyW8A8Conv2dProtected
from torch.nn.modules.linear import Linear
from torch.nn.modules.conv import Conv2d
import pdb
import re
import torch.multiprocessing as mp
import torch
import sys
from pathlib import Path
from InjectionState import _injection_state
import torch.nn as nn
from concurrent.futures import ThreadPoolExecutor, as_completed
def _build_noisy_linear_task(args):
    """在子线程中执行权重复制 + 新模块创建"""
    (
        original_module, protected, weight_quant, act_quant,
        quantize_bmm_input, err_prob, err_fn, method
    ) = args

    if not protected:
        return NoisyW8A8Linear.from_float(
            original_module,
            weight_quant=weight_quant,
            act_quant=act_quant,
            quantize_output=quantize_bmm_input,
            err_prob=err_prob,
            err_fn=err_fn
        )
    else:
        return NoisyW8A8LinearProtected.from_float(
            original_module,
            weight_quant=weight_quant,
            act_quant=act_quant,
            quantize_output=quantize_bmm_input,
            err_prob=err_prob,
            method=method,
            err_fn=err_fn
        )


def resolve_module(root, path):
    """
    在 root（通常是 self.model 或 self）上根据 path 查找并返回目标 module。
    支持: foo[3] / foo.3 / foo.bar
    """
    if not path:
        return root

    # 去掉可能的 self. / model. 前缀
    path = re.sub(r'^(self\.|model\.)', '', path)

    # 统一把 [3] 变成 .3
    path = re.sub(r'\[(\-?\d+)\]', r'.\1', path)

    cur = root
    for token in path.split('.'):
        if token == '':
            continue
        if re.fullmatch(r'-?\d+', token):
            idx = int(token)
            cur = cur[idx]
        else:
            if hasattr(cur, token):
                cur = getattr(cur, token)
            elif isinstance(cur, nn.Module) and token in cur._modules:
                cur = cur._modules[token]
            else:
                raise AttributeError(f"Module has no attribute or submodule '{token}' on {cur}")
    return cur


def create_noisy_module(orig_module, protected, weight_quant, act_quant,
                        quantize_bmm_input, err_prob, method, err_fn):
    if not protected:
        return NoisyW8A8Linear.from_float(
            orig_module,
            weight_quant=weight_quant,
            act_quant=act_quant,
            quantize_output=quantize_bmm_input,
            err_prob=err_prob,
            err_fn=err_fn
        )
    else:
        return NoisyW8A8LinearProtected.from_float(
            orig_module,
            weight_quant=weight_quant,
            act_quant=act_quant,
            quantize_output=quantize_bmm_input,
            err_prob=err_prob,
            method=method,
            err_fn=err_fn
        )


def parse_target(target: str, err_prob: float, modules_for_select: dict, default_modules: list = None):
    """
    支持 step 格式：
      - 单步：-step12
      - 区间：-step5t15
      - 下界：-step5t  表示 step >=5
      - 多段：-step5t15-step25t45

    return modules_select(list of tuples), err_fn(current_step) -> float
    """
    # 解析所有 step 区间
    step_ranges = []
    for step_match in re.finditer(r'-step(\d+(?:t\d*)?)', target):
        step_str = step_match.group(1)
        if 't' in step_str:
            start_str, end_str = step_str.split('t')
            start = int(start_str)
            end = int(end_str) if end_str else None
            step_ranges.append((start, end))
        else:
            val = int(step_str)
            step_ranges.append((val, val))
    
    # 解析 modules
    all_flag = False
    modules_match = re.search(r'-modules_([A-Za-z0-9_]+)', target)
    if modules_match:
        modules = modules_match.group(1).split('_')
        if "all" in modules:
            all_flag = True
            print("all_flag is set to True, 这可能会导致运行非常慢.")
        else:
            invalid = [m for m in modules if m not in modules_for_select]
            if invalid:
                raise ValueError(f"target 中包含非法模块名: {invalid}")
    elif default_modules is not None:
        invalid = [m for m in default_modules if m not in modules_for_select]
        if invalid:
            raise ValueError(f"default_modules 中包含非法模块名: {invalid}")
        modules = default_modules
    else:
        modules = list(modules_for_select.keys())

    # 构造 err_fn
    if step_ranges:
        def err_fn(current_step):
            for start, end in step_ranges:
                if end is None:
                    if current_step >= start:
                        return err_prob
                else:
                    if start <= current_step <= end:
                        return err_prob
            return 0.0
    else:
        # 保持原来行为：未指定 step 时，对所有有效 step（>=0）返回 err_prob
        def err_fn(current_step):
            return err_prob if current_step >= 0 else 0.0
    
    # 构造 modules_select
    if all_flag:
        modules_select = -1  # 表示所有模块
    else:
        modules_select = [modules_for_select[m] for m in modules]

    return modules_select, err_fn


def print_activations(module, input, output):
    with open('output.txt', 'a') as outfile:
        print(f"Layer: {module.__class__.__name__}", file=outfile)
        print(f"Output (activations): {output}\n", file=outfile)

def modify_llama_attention(module, weight_quant, act_quant, quantize_bmm_input, err_prob, protected=False, method='none'):
    if protected:
        module.q_proj = NoisyW8A8LinearProtected.from_float(module.q_proj, weight_quant=weight_quant, 
                    act_quant=act_quant, quantize_output=quantize_bmm_input, err_prob=err_prob, method=method)
        module.k_proj = NoisyW8A8LinearProtected.from_float(module.k_proj, weight_quant=weight_quant, 
                    act_quant=act_quant, quantize_output=quantize_bmm_input, err_prob=err_prob, method=method)
        module.v_proj = NoisyW8A8LinearProtected.from_float(module.v_proj, weight_quant=weight_quant, 
                    act_quant=act_quant, quantize_output=quantize_bmm_input, err_prob=err_prob, method=method)
        module.o_proj = NoisyW8A8LinearProtected.from_float(module.o_proj, weight_quant=weight_quant, 
                    act_quant=act_quant, quantize_output=False, err_prob=err_prob, method=method)
    else:
        module.q_proj = NoisyW8A8Linear.from_float(module.q_proj, weight_quant=weight_quant, 
            act_quant=act_quant, quantize_output=quantize_bmm_input, err_prob=err_prob)
        module.k_proj = NoisyW8A8Linear.from_float(module.k_proj, weight_quant=weight_quant, 
                    act_quant=act_quant, quantize_output=quantize_bmm_input, err_prob=err_prob)
        module.v_proj = NoisyW8A8Linear.from_float(module.v_proj, weight_quant=weight_quant, 
                    act_quant=act_quant, quantize_output=quantize_bmm_input, err_prob=err_prob)
        module.o_proj = NoisyW8A8Linear.from_float(module.o_proj, weight_quant=weight_quant, 
                    act_quant=act_quant, quantize_output=False, err_prob=err_prob)
        
    module.to('cpu')
    return module


def modify_llama_attention_k_proj(module, weight_quant, act_quant, quantize_bmm_input, err_prob, protected=False, method='none'):
    if protected:
        module.k_proj = NoisyW8A8LinearProtected.from_float(module.k_proj, weight_quant=weight_quant, act_quant=act_quant, quantize_output=quantize_bmm_input, err_prob=err_prob, method=method)
    else:
        module.k_proj = NoisyW8A8Linear.from_float(module.k_proj, weight_quant=weight_quant, 
                act_quant=act_quant, quantize_output=quantize_bmm_input, err_prob=err_prob)
    module.to('cpu')
    return module

def modify_llama_attention_o_proj(module, weight_quant, act_quant, quantize_bmm_input, err_prob, protected=False, method='none'):
    if protected:
        module.o_proj = NoisyW8A8LinearProtected.from_float(module.o_proj, weight_quant=weight_quant, act_quant=act_quant, quantize_output=quantize_bmm_input, err_prob=err_prob, method=method)
    else:
        module.o_proj = NoisyW8A8Linear.from_float(module.o_proj, weight_quant=weight_quant, 
                act_quant=act_quant, quantize_output=quantize_bmm_input, err_prob=err_prob)
    module.to('cpu')
    return module

def modify_clip_attention(module, weight_quant, act_quant, quantize_bmm_input, err_prob, protected=False, method='none'):
    if protected:
        module.q_proj = NoisyW8A8LinearProtected.from_float(module.q_proj, weight_quant=weight_quant, 
                    act_quant=act_quant, quantize_output=quantize_bmm_input, err_prob=err_prob, method=method)
        module.k_proj = NoisyW8A8LinearProtected.from_float(module.k_proj, weight_quant=weight_quant, 
                    act_quant=act_quant, quantize_output=quantize_bmm_input, err_prob=err_prob, method=method)
        module.v_proj = NoisyW8A8LinearProtected.from_float(module.v_proj, weight_quant=weight_quant, 
                    act_quant=act_quant, quantize_output=quantize_bmm_input, err_prob=err_prob, method=method)
        module.out_proj = NoisyW8A8LinearProtected.from_float(module.out_proj, weight_quant=weight_quant, 
                    act_quant=act_quant, quantize_output=False, err_prob=err_prob, method=method)
    else:
        module.q_proj = NoisyW8A8Linear.from_float(module.q_proj, weight_quant=weight_quant, 
                    act_quant=act_quant, quantize_output=quantize_bmm_input, err_prob=err_prob)
        module.k_proj = NoisyW8A8Linear.from_float(module.k_proj, weight_quant=weight_quant, 
                    act_quant=act_quant, quantize_output=quantize_bmm_input, err_prob=err_prob)
        module.v_proj = NoisyW8A8Linear.from_float(module.v_proj, weight_quant=weight_quant, 
                    act_quant=act_quant, quantize_output=quantize_bmm_input, err_prob=err_prob)
        module.out_proj = NoisyW8A8Linear.from_float(module.out_proj, weight_quant=weight_quant, 
                    act_quant=act_quant, quantize_output=False, err_prob=err_prob)
    module.to('cpu')
    return module
    
def modify_llama_mlp(module, weight_quant, act_quant, err_prob, protected=False, method='none'):
    if protected:
        module.gate_proj = NoisyW8A8LinearProtected.from_float(module.gate_proj, weight_quant=weight_quant,
            act_quant=act_quant, quantize_output=False, err_prob=err_prob, method=method)
        module.up_proj = NoisyW8A8LinearProtected.from_float(module.up_proj, weight_quant=weight_quant,
            act_quant=act_quant, quantize_output=False, err_prob=err_prob, method=method)
        module.down_proj = NoisyW8A8LinearProtected.from_float(module.down_proj, weight_quant=weight_quant,
            act_quant=act_quant, quantize_output=False, err_prob=err_prob, method=method)
    else:
        module.gate_proj = NoisyW8A8Linear.from_float(module.gate_proj, weight_quant=weight_quant,
            act_quant=act_quant, quantize_output=False, err_prob=err_prob)
        module.up_proj = NoisyW8A8Linear.from_float(module.up_proj, weight_quant=weight_quant,
            act_quant=act_quant, quantize_output=False, err_prob=err_prob)
        module.down_proj = NoisyW8A8Linear.from_float(module.down_proj, weight_quant=weight_quant,
            act_quant=act_quant, quantize_output=False, err_prob=err_prob)
    module.to('cpu')
    return module
    
    
def modify_clip_mlp(module, weight_quant, act_quant, err_prob, protected=False, method='none'):
    if protected:
        module.fc1 = NoisyW8A8LinearProtected.from_float(module.fc1, weight_quant=weight_quant,
            act_quant=act_quant, quantize_output=False, err_prob=err_prob, method=method)
        module.fc2 = NoisyW8A8LinearProtected.from_float(module.fc2, weight_quant=weight_quant,
            act_quant=act_quant, quantize_output=False, err_prob=err_prob, method=method)
    else:
        module.fc1 = NoisyW8A8Linear.from_float(module.fc1, weight_quant=weight_quant,
            act_quant=act_quant, quantize_output=False, err_prob=err_prob)
        module.fc2 = NoisyW8A8Linear.from_float(module.fc2, weight_quant=weight_quant,
            act_quant=act_quant, quantize_output=False, err_prob=err_prob)
    module.to('cpu')
    return module


class FIassistant:
    def __init__(self, model):
        self.model = model
    def replace_modules_in_parallel(self, target_module_paths, protected,
                                weight_quant, act_quant, quantize_bmm_input,
                                err_prob, method, err_fn, max_workers=16):
        """并行构造替换模块，加速注错注入过程"""

        # 1️⃣ 收集任务
        replace_tasks = []
        parent_refs = []

        for parent_path, module_name in target_module_paths:
            try:
                parent = eval(f"self.{parent_path}")
            except Exception as e:
                print(f"无法解析 parent_path={parent_path}: {e}")
                raise TypeError(f"Unsupported module type for {parent_path}.{module_name}: {type(original_module)}")

            try:
                if re.fullmatch(r'-?\d+', str(module_name)):
                    idx = int(module_name)
                    original_module = parent[idx]
                else:
                    original_module = getattr(parent, module_name)
            except Exception as e:
                print(f"无法访问 module={module_name}: {e}")
                raise TypeError(f"Unsupported module type for {parent_path}.{module_name}: {type(original_module)}")

            if isinstance(original_module, nn.Linear):
                replace_tasks.append((
                    original_module, protected, weight_quant, act_quant,
                    quantize_bmm_input, err_prob, err_fn, method
                ))
                parent_refs.append((parent, module_name, None))
            elif isinstance(original_module, nn.ModuleList):
                # 替换其中第一个 Linear
                for idx, m in enumerate(original_module):
                    if isinstance(m, nn.Linear):
                        replace_tasks.append((
                            m, protected, weight_quant, act_quant,
                            quantize_bmm_input, err_prob, err_fn, method
                        ))
                        parent_refs.append((original_module, idx, "list"))
                        break

        # 2️⃣ 并行执行替换构造
        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(_build_noisy_linear_task, task): i for i, task in enumerate(replace_tasks)}
            for fut in as_completed(futures):
                idx = futures[fut]
                try:
                    new_mod = fut.result()
                    results.append((idx, new_mod))
                except Exception as e:
                    import traceback
                    print(f"[Error] 并行构造模块失败: {e}")
                    traceback.print_exc()

        # 3️⃣ 写回主线程（安全操作）
        for (idx, new_mod) in results:
            parent, module_name, mtype = parent_refs[idx]
            if mtype == "list":
                parent[module_name] = new_mod
            elif re.fullmatch(r'-?\d+', str(module_name)):
                parent[int(module_name)] = new_mod
            else:
                setattr(parent, module_name, new_mod)
        
    @staticmethod
    def is_target_layer(layer_name, target_layers):
        # 如果目标层列表为空，返回 True
        if not target_layers:
            return True
        match = re.search(r'\.(\d+)\.', layer_name)
        if match:
            first_layer_number = int(match.group(1))
            return (first_layer_number in target_layers)
        return False

    def clear_all_noisy_linear_caches(self):
        """
        清空模型中所有NoisyW8A8Linear层的缓存
        """
        cleared_count = 0
        
        for name, module in self.model.named_modules():
            if isinstance(module, NoisyW8A8Linear):
                if hasattr(module, 'cache') and module.cache is not None:
                    module.cache = None
                    cleared_count += 1
                    # print(f"Cleared cache in {name}")
        print(f"Cleared caches in {cleared_count} NoisyW8A8Linear layers")
        return cleared_count

    def inject_fault_to_module(self, target="", weight_quant='per_channel', act_quant='per_token', quantize_bmm_input=True, err_prob=0.0, target_layers=[], protected=False, method='AD'):
        

        if target == 'DiT_Linear':
            # 存储目标模块及其父模块的路径
            target_module_paths = [
                ('model.action_model.net.history_embedder', 'linear'),
                ('model.action_model.net.x_embedder', 'linear'),
                ('model.action_model.net.t_embedder.mlp', '0'),
                ('model.action_model.net.t_embedder.mlp', '2'),
                ('model.action_model.net.z_embedder', 'linear'),
                ('model.action_model.net.final_layer', 'linear'),
            ]
            for i in [0,6,11]:
                target_module_paths.append((f'model.action_model.net.blocks[{i}].attn', 'qkv'))
                target_module_paths.append((f'model.action_model.net.blocks[{i}].attn', 'proj'))
                target_module_paths.append((f'model.action_model.net.blocks[{i}].mlp', 'fc1'))
                target_module_paths.append((f'model.action_model.net.blocks[{i}].mlp', 'fc2'))

            for parent_path, module_name in target_module_paths:
                # 获取父模块
                parent = eval(f'self.{parent_path}')
                # 获取原始模块
                original_module = getattr(parent, module_name)
                
                # 替换模块
                if not protected:
                    new_module = NoisyW8A8Linear.from_float(
                        original_module, 
                        weight_quant=weight_quant,  
                        act_quant=act_quant, 
                        quantize_output=quantize_bmm_input, 
                        err_prob=err_prob
                    )
                else:
                    new_module = NoisyW8A8LinearProtected.from_float(
                        original_module, 
                        weight_quant=weight_quant,  
                        act_quant=act_quant, 
                        quantize_output=quantize_bmm_input, 
                        err_prob=err_prob, 
                        method=method
                    )
                
                # 直接替换父模块的属性
                setattr(parent, module_name, new_module)

        elif target.startswith("DiT_diffusion"):  # eg "DiT_diffusion5"
            # 提取后半部分（去除"DiT_diffusion"）
            suffix = target[len("DiT_diffusion"):]  # 切片从前缀长度开始
            if suffix:  # 有具体数字后缀
                concrete_step = int(suffix)
                err_fn = lambda step, cs=concrete_step, ep=err_prob: ep if step == cs else 0.0
            else:  # 没有后缀，则在step>=0返回 err_prob, step < 0 表示profile运行
                err_fn = lambda step, ep=err_prob: ep if step >= 0 else 0
            #err_fn = lambda step, ep=err_prob: ep
            target_module_paths = [
                ('model.action_model.net.history_embedder', 'linear'),
                ('model.action_model.net.x_embedder', 'linear'),
                ('model.action_model.net.t_embedder.mlp', '0'),
                ('model.action_model.net.t_embedder.mlp', '2'),
                ('model.action_model.net.z_embedder', 'linear'),
                ('model.action_model.net.final_layer', 'linear'),
            ]
            for i in [0,6,11]:
                target_module_paths.append((f'model.action_model.net.blocks[{i}].attn', 'qkv'))
                target_module_paths.append((f'model.action_model.net.blocks[{i}].attn', 'proj'))
                target_module_paths.append((f'model.action_model.net.blocks[{i}].mlp', 'fc1'))
                target_module_paths.append((f'model.action_model.net.blocks[{i}].mlp', 'fc2'))

            for parent_path, module_name in target_module_paths:
                # 获取父模块
                parent = eval(f'self.{parent_path}')
                # 获取原始模块
                original_module = getattr(parent, module_name)
                
                # 替换模块
                if not protected:
                    new_module = NoisyW8A8Linear.from_float(
                        original_module, 
                        weight_quant=weight_quant,  
                        act_quant=act_quant, 
                        quantize_output=quantize_bmm_input, 
                        err_prob=err_prob,
                        err_fn = err_fn
                    )
                else:
                    new_module = NoisyW8A8LinearProtected.from_float(
                        original_module, 
                        weight_quant=weight_quant,  
                        act_quant=act_quant, 
                        quantize_output=quantize_bmm_input, 
                        err_prob=err_prob, 
                        method=method,
                        err_fn = err_fn
                    )
                
                # 直接替换父模块的属性
                setattr(parent, module_name, new_module)
        
        elif target == "octo-linear":
            block = self.model.module.octo_transformer.task_tokenizers.language.hf_model.encoder.block[9]
            for name, module in block.named_modules():
                if isinstance(module, torch.nn.Linear):
                    # 直接操作 block 的子模块
                    parts = name.split('.')
                    parent = block
                    for part in parts[:-1]:  # 逐级找到父模块
                        parent = getattr(parent, part)
                    
                    if not protected:
                        new_module = NoisyW8A8Linear.from_float(
                            module, 
                            weight_quant=weight_quant,  
                            act_quant=act_quant, 
                            quantize_output=quantize_bmm_input, 
                            err_prob=err_prob
                        )
                    else:
                        new_module = NoisyW8A8LinearProtected.from_float(
                            module, 
                            weight_quant=weight_quant,  
                            act_quant=act_quant, 
                            quantize_output=quantize_bmm_input, 
                            err_prob=err_prob, 
                            method=method
                        )
                    
                    # 替换父模块的子模块
                    setattr(parent, parts[-1], new_module)

        #### benchmark1
        elif target.startswith("DiTXL512"):
            if not hasattr(self, "model"):
                raise RuntimeError("self.model not found for DiT-XL512 injection")
            # 参数验证
            if (target_layers is not None and (target_layers != "")) and (not all(isinstance(x, int) for x in target_layers) or not isinstance(target_layers, (list, tuple))):
                raise TypeError("target_layers must be a list or tuple of integers")
            
            modules_for_select = {}
            default_modules = []
            
            # === Helper function to register one module ===
            def _add_module(key, parent_path, module_name, should_inject):
                """记录一个待替换模块"""
                modules_for_select[key] = (parent_path, module_name, should_inject)
                default_modules.append(key)

            # === Step 1: 构造所有模块 ===
            n_blocks = len(self.model.blocks)
            target_set = set(target_layers or range(n_blocks))
            
            print(f"[DiT-XL512 Injection] Total blocks: {n_blocks}, Target layers: {sorted(list(target_set))}")
        
            for i in range(n_blocks): 
                inject_flag = (i in target_set) and ("full" in target) #注意即使injecct_flag为False也是要做量化的
                 
                # Attention 模块
                attn_path = f"model.blocks[{i}].attn"
                _add_module(f"block{i}_attn_qkv", attn_path, "qkv", inject_flag)
                _add_module(f"block{i}_attn_proj", attn_path, "proj", inject_flag)
                
                # MLP 模块  
                mlp_path = f"model.blocks[{i}].mlp"
                _add_module(f"block{i}_mlp_fc1", mlp_path, "fc1", inject_flag)
                _add_module(f"block{i}_mlp_fc2", mlp_path, "fc2", inject_flag)

            # 处理 embedding 相关模块
            if "emb" in target:
                embed_inject_flag = True
                # Time embedder mlp 层
                temb_path = "model.t_embedder.mlp"
                _add_module("t_embedder_mlp_0", temb_path, "0", embed_inject_flag)
                _add_module("t_embedder_mlp_2", temb_path, "2", embed_inject_flag)

            # === Step 2: 替换所有模块为 NoisyW8A8Linear (err_prob=0) ===
            all_module_paths = [(p, m) for (p, m, _) in modules_for_select.values()]
            print(f"[DiT-XL512 Injection] Replacing {len(all_module_paths)} Linear modules with NoisyW8A8Linear...")

            self.replace_modules_in_parallel(
                target_module_paths=all_module_paths,
                protected=protected,
                weight_quant=weight_quant,
                act_quant=act_quant,
                quantize_bmm_input=quantize_bmm_input,
                err_prob=0.0,        # 默认无错
                err_fn=lambda x: 0.0,   # 默认无错函数
                method=method
            )

            # === Step 3: 为选中的模块生成 err_fn ===
            _, err_fn = parse_target(target, err_prob, modules_for_select, default_modules)

            # === Step 4: 激活目标层的错误注入 ===
            activated = 0
            for key, (parent_path, module_name, should_inject) in modules_for_select.items():
                if should_inject:
                    try:
                        parent = eval(f"self.{parent_path}")
                        mod = getattr(parent, module_name)
                        mod.err_prob = err_prob
                        mod.err_fn = err_fn
                        activated += 1
                    except Exception as e:
                        print(f"[Warning] Failed to activate err_fn for {key}: {e}")
                        raise e

            print(f"[DiT-XL512 Injection] Activated error injection for {activated}/{len(all_module_paths)} modules ✅")
            print(self.model)


        elif target.startswith("Flux_dev1"):
            if not hasattr(self, "model"): 
                raise RuntimeError("self.model not found for FLUX-dev1 injection") 
            modules_for_select = {} 
            default_modules = [] 

            if target.startswith("Flux_dev1_full"):
                # ========== transformer_blocks 注入点 ==========
                n_tb = len(self.model.transformer_blocks) 
                for i in range(0, n_tb, 1): 
                    # Attention 模块中的关键Linear层
                    key_to_q = f"t{i}_to_q"
                    parent_path_to_q = f"model.transformer_blocks[{i}].attn"
                    modules_for_select[key_to_q] = (parent_path_to_q, "to_q")
                    default_modules.append(key_to_q)
                    
                    key_to_k = f"t{i}_to_k" 
                    parent_path_to_k = f"model.transformer_blocks[{i}].attn"
                    modules_for_select[key_to_k] = (parent_path_to_k, "to_k")
                    default_modules.append(key_to_k)
                    
                    key_to_v = f"t{i}_to_v"
                    parent_path_to_v = f"model.transformer_blocks[{i}].attn"
                    modules_for_select[key_to_v] = (parent_path_to_v, "to_v")
                    default_modules.append(key_to_v)
                    
                    # to_out 和 to_add_out (原有的)
                    key_out = f"t{i}_to_out" 
                    parent_path_out = f"model.transformer_blocks[{i}].attn.to_out" 
                    modules_for_select[key_out] = (parent_path_out, "0") 
                    default_modules.append(key_out) 
                    
                    key_add = f"t{i}_to_add_out" 
                    parent_path_add = f"model.transformer_blocks[{i}].attn" 
                    modules_for_select[key_add] = (parent_path_add, "to_add_out") 
                    default_modules.append(key_add)
                    
                    # 额外的 q/k/v projection
                    key_add_q = f"t{i}_add_q_proj"
                    parent_path_add_q = f"model.transformer_blocks[{i}].attn"
                    modules_for_select[key_add_q] = (parent_path_add_q, "add_q_proj")
                    default_modules.append(key_add_q)
                    
                    key_add_k = f"t{i}_add_k_proj"
                    parent_path_add_k = f"model.transformer_blocks[{i}].attn"
                    modules_for_select[key_add_k] = (parent_path_add_k, "add_k_proj")
                    default_modules.append(key_add_k)
                    
                    key_add_v = f"t{i}_add_v_proj"
                    parent_path_add_v = f"model.transformer_blocks[{i}].attn"
                    modules_for_select[key_add_v] = (parent_path_add_v, "add_v_proj")
                    default_modules.append(key_add_v)
                    
                    # FeedForward 模块中的Linear层
                    key_ff1 = f"t{i}_ff_0_proj"
                    parent_path_ff1 = f"model.transformer_blocks[{i}].ff.net"
                    modules_for_select[key_ff1] = (parent_path_ff1, "0.proj")
                    default_modules.append(key_ff1)
                    
                    key_ff2 = f"t{i}_ff_2"
                    parent_path_ff2 = f"model.transformer_blocks[{i}].ff.net"
                    modules_for_select[key_ff2] = (parent_path_ff2, "2")
                    default_modules.append(key_ff2)
                    
                    # Context FeedForward 模块中的Linear层
                    key_ff_ctx1 = f"t{i}_ff_ctx_0_proj"
                    parent_path_ff_ctx1 = f"model.transformer_blocks[{i}].ff_context.net"
                    modules_for_select[key_ff_ctx1] = (parent_path_ff_ctx1, "0.proj")
                    default_modules.append(key_ff_ctx1)
                    
                    key_ff_ctx2 = f"t{i}_ff_ctx_2"
                    parent_path_ff_ctx2 = f"model.transformer_blocks[{i}].ff_context.net"
                    modules_for_select[key_ff_ctx2] = (parent_path_ff_ctx2, "2")
                    default_modules.append(key_ff_ctx2)
                    
                
                # ========== single_transformer_blocks 注入点 ==========
                n_sb = len(self.model.single_transformer_blocks) 
                for i in range(0, n_sb, 1): 
                    # 原有的 proj_out
                    key_proj = f"s{i}_proj_out" 
                    parent_path_proj = f"model.single_transformer_blocks[{i}]" 
                    modules_for_select[key_proj] = (parent_path_proj, "proj_out") 
                    default_modules.append(key_proj)
                    
                    # Attention 模块中的Linear层
                    key_s_to_q = f"s{i}_to_q"
                    parent_path_s_to_q = f"model.single_transformer_blocks[{i}].attn"
                    modules_for_select[key_s_to_q] = (parent_path_s_to_q, "to_q")
                    default_modules.append(key_s_to_q)
                    
                    key_s_to_k = f"s{i}_to_k"
                    parent_path_s_to_k = f"model.single_transformer_blocks[{i}].attn"
                    modules_for_select[key_s_to_k] = (parent_path_s_to_k, "to_k")
                    default_modules.append(key_s_to_k)
                    
                    key_s_to_v = f"s{i}_to_v"
                    parent_path_s_to_v = f"model.single_transformer_blocks[{i}].attn"
                    modules_for_select[key_s_to_v] = (parent_path_s_to_v, "to_v")
                    default_modules.append(key_s_to_v)
                    
                    # MLP 相关的Linear层
                    key_proj_mlp = f"s{i}_proj_mlp"
                    parent_path_proj_mlp = f"model.single_transformer_blocks[{i}]"
                    modules_for_select[key_proj_mlp] = (parent_path_proj_mlp, "proj_mlp")
                    default_modules.append(key_proj_mlp)

            else:
                # transformer_blocks 每个 block 注错 to_out 和 to_add_out 
                n_tb = len(self.model.transformer_blocks) 
                for i in range(0, n_tb, 1): 
                    key_out = f"t{i}_to_out" 
                    parent_path_out = f"model.transformer_blocks[{i}].attn.to_out" 
                    modules_for_select[key_out] = (parent_path_out, "0") 
                    default_modules.append(key_out) 
                    key_add = f"t{i}_to_add_out" 
                    parent_path_add = f"model.transformer_blocks[{i}].attn" 
                    modules_for_select[key_add] = (parent_path_add, "to_add_out") 
                    default_modules.append(key_add) 
                # single_transformer_blocks 每个 block 注错 proj_out 
                n_sb = len(self.model.single_transformer_blocks) 
                for i in range(0, n_sb, 1): 
                    key_proj = f"s{i}_proj_out" 
                    parent_path_proj = f"model.single_transformer_blocks[{i}]" 
                    modules_for_select[key_proj] = (parent_path_proj, "proj_out") 
                    default_modules.append(key_proj) 

            # 调用 parse_target 处理 step 参数，得到 target_module_paths 和 err_fn 
            target_module_paths, err_fn = parse_target(target, err_prob, modules_for_select, default_modules) 
            # 调用你写好的 replace_modules_in_parallel 统一处理替换逻辑 
            self.replace_modules_in_parallel( 
                target_module_paths=target_module_paths, 
                protected=protected, 
                weight_quant=weight_quant, 
                act_quant=act_quant, 
                quantize_bmm_input=quantize_bmm_input,
                err_prob=err_prob, 
                err_fn=err_fn, 
                method=method ) 
            print("[Flux_dev1] Module injection complete ✅")

        elif target.startswith("Exp_Flux_dev1"):
            if not hasattr(self, "model"):
                raise RuntimeError("self.model not found for Exp_Flux_dev1 injection")

            modules_for_select = {}
            default_modules = []

            # ------------------------------
            # Layer single: FluxSingleTransformerBlock 所有 linear 层
            # ------------------------------
            if target.startswith("Exp_Flux_dev1_layer_single"):
                for i in target_layers:
                    block = f"model.single_transformer_blocks[{i}]"
                    # proj_mlp / act_mlp / proj_out are attributes on the block itself
                    modules_for_select[f"s{i}_proj_mlp"] = (block, "proj_mlp")
                    modules_for_select[f"s{i}_act_mlp"] = (block, "act_mlp")
                    modules_for_select[f"s{i}_proj_out"] = (block, "proj_out")
                    # attn.* are attributes under block.attn
                    modules_for_select[f"s{i}_attn_to_q"] = (f"{block}.attn", "to_q")
                    modules_for_select[f"s{i}_attn_to_k"] = (f"{block}.attn", "to_k")
                    modules_for_select[f"s{i}_attn_to_v"] = (f"{block}.attn", "to_v")
                    # collect defaults
                    default_modules.append(f"s{i}_proj_mlp")
                    default_modules.append(f"s{i}_act_mlp")
                    default_modules.append(f"s{i}_proj_out")
                    default_modules.append(f"s{i}_attn_to_q")
                    default_modules.append(f"s{i}_attn_to_k")
                    default_modules.append(f"s{i}_attn_to_v")

            # ------------------------------
            # Layer double: FluxTransformerBlock 所有 linear 层
            # ------------------------------
            elif target.startswith("Exp_Flux_dev1_layer_double"):
                for i in target_layers:
                    block = f"model.transformer_blocks[{i}]"
                    modules_for_select[f"t{i}_ff_0_proj"] = (f"{block}.ff.net[0]", "proj")  # if ff.net[0] itself has .proj attribute
                    modules_for_select[f"t{i}_ff_2"] = (f"{block}.ff.net", "2")
                    # attn linear 层
                    modules_for_select[f"t{i}_attn_to_q"] = (f"{block}.attn", "to_q")
                    modules_for_select[f"t{i}_attn_to_k"] = (f"{block}.attn", "to_k")
                    modules_for_select[f"t{i}_attn_to_v"] = (f"{block}.attn", "to_v")
                    # to_out 是个 list/ModuleList 通常，取第0项 -> parent: attn.to_out, module_name: "0"
                    modules_for_select[f"t{i}_attn_to_out_0"] = (f"{block}.attn.to_out", "0")
                    default_modules.append(f"t{i}_ff_0_proj")
                    default_modules.append(f"t{i}_ff_2")
                    default_modules.append(f"t{i}_attn_to_q")
                    default_modules.append(f"t{i}_attn_to_k")
                    default_modules.append(f"t{i}_attn_to_v")
                    default_modules.append(f"t{i}_attn_to_out_0")

            # ------------------------------
            # DTO / DADD: FluxTransformerBlock attn 层
            # ------------------------------
            elif target.startswith("Exp_Flux_dev1_layer_dto"):
                for i in target_layers:
                    block = f"model.transformer_blocks[{i}]"
                    modules_for_select[f"t{i}_to_q"] = (f"{block}.attn", "to_q")
                    modules_for_select[f"t{i}_to_k"] = (f"{block}.attn", "to_k")
                    modules_for_select[f"t{i}_to_v"] = (f"{block}.attn", "to_v")
                    modules_for_select[f"t{i}_to_out"] = (f"{block}.attn.to_out", "0")  # use index "0"
                    default_modules.append(f"t{i}_to_q")
                    default_modules.append(f"t{i}_to_k")
                    default_modules.append(f"t{i}_to_v")
                    default_modules.append(f"t{i}_to_out")

            elif target.startswith("Exp_Flux_dev1_layer_dadd"):
                for i in target_layers:
                    block = f"model.transformer_blocks[{i}]"
                    modules_for_select[f"t{i}_add_q_proj"] = (f"{block}.attn", "add_q_proj")
                    modules_for_select[f"t{i}_add_k_proj"] = (f"{block}.attn", "add_k_proj")
                    modules_for_select[f"t{i}_add_v_proj"] = (f"{block}.attn", "add_v_proj")
                    modules_for_select[f"t{i}_to_add_out"] = (f"{block}.attn", "to_add_out")
                    default_modules.append(f"t{i}_add_q_proj")
                    default_modules.append(f"t{i}_add_k_proj")
                    default_modules.append(f"t{i}_add_v_proj")
                    default_modules.append(f"t{i}_to_add_out")

            # ------------------------------
            # Embed 分支
            # ------------------------------
            elif target.startswith("Exp_Flux_dev1_embed"):
                # time embed
                if target.startswith("Exp_Flux_dev1_embed_time"):
                    modules_for_select["timestep_embedder_linear_1"] = ("model.time_text_embed.timestep_embedder", "linear_1")
                    modules_for_select["timestep_embedder_linear_2"] = ("model.time_text_embed.timestep_embedder", "linear_2")
                    default_modules.append("timestep_embedder_linear_1")
                    default_modules.append("timestep_embedder_linear_2")

                # guidance embed
                elif target.startswith("Exp_Flux_dev1_embed_time_guide"):
                    modules_for_select["guidance_embedder_linear_1"] = ("model.time_text_embed.guidance_embedder", "linear_1")
                    modules_for_select["guidance_embedder_linear_2"] = ("model.time_text_embed.guidance_embedder", "linear_2")
                    default_modules.append("guidance_embedder_linear_1")
                    default_modules.append("guidance_embedder_linear_2")

                # text embed
                elif target.startswith("Exp_Flux_dev1_embed_text"):
                    modules_for_select["text_embedder_linear_1"] = ("model.time_text_embed.text_embedder", "linear_1")
                    modules_for_select["text_embedder_linear_2"] = ("model.time_text_embed.text_embedder", "linear_2")
                    default_modules.append("text_embedder_linear_1")
                    default_modules.append("text_embedder_linear_2")

                else:
                    raise RuntimeError(f"Unknown Exp_Flux_dev1_embed target: {target}")

            else:
                raise RuntimeError(f"Unknown Exp_Flux_dev1 target: {target}")

            # ------------------------------
            # 统一调用 parse_target + replace_modules_in_parallel
            # ------------------------------
            target_module_paths, err_fn = parse_target(target, err_prob, modules_for_select, default_modules)

            self.replace_modules_in_parallel(
                target_module_paths=target_module_paths,
                protected=protected,
                weight_quant=weight_quant,
                act_quant=act_quant,
                quantize_bmm_input=quantize_bmm_input,
                err_prob=err_prob,
                err_fn=err_fn,
                method=method
            )

            print("#########################################################################")
            print(f"[Exp_Flux_dev1] Module injection {target} complete ✅")
            print(self.model)

        ### benchmark2
        elif target.startswith("PixArt"):
            if not hasattr(self, "model"):
                raise RuntimeError("self.model not found for PixArt injection")
            ####先对整体做无错量化
            modules_for_select = {}
            default_modules = []
            if (target_layers is not None and (target_layers != "")) and (not all(isinstance(x, int) for x in target_layers) or not isinstance(target_layers, (list, tuple))):
                raise TypeError("target_layers must be a list or tuple of integers")
            n_blocks = len(self.model.transformer_blocks)
            target_set = set(target_layers or range(n_blocks))
    
            print(f"[PixArt Injection] Total blocks: {n_blocks}, Target layers: {sorted(list(target_set))}")

            # === Helper function to register one module ===
            def _add_module(key, parent_path, module_name, should_inject):
                """记录一个待替换模块"""
                modules_for_select[key] = (parent_path, module_name, should_inject)
                default_modules.append(key)

            # === Step 1: 构造所有模块 ===
            for i in range(n_blocks):
                inject_flag = (i in target_set) and ("full" in target)  ###类似PixArtemb的时候非embedding部分应该inject_flag取False

                # ---- Attention 1 ----
                attn1 = f"model.transformer_blocks[{i}].attn1"
                _add_module(f"block{i}_attn1_to_q", attn1, "to_q", inject_flag)
                _add_module(f"block{i}_attn1_to_k", attn1, "to_k", inject_flag)
                _add_module(f"block{i}_attn1_to_v", attn1, "to_v", inject_flag)
                _add_module(f"block{i}_attn1_to_out", attn1 + ".to_out", "0", inject_flag)

                # ---- Attention 2 ----
                attn2 = f"model.transformer_blocks[{i}].attn2"
                _add_module(f"block{i}_attn2_to_k", attn2, "to_k", inject_flag)
                _add_module(f"block{i}_attn2_to_v", attn2, "to_v", inject_flag)
                _add_module(f"block{i}_attn2_to_out", attn2 + ".to_out", "0", inject_flag)

                # ---- FeedForward ----
                ff = f"model.transformer_blocks[{i}].ff.net"
                _add_module(f"block{i}_ff_proj1", ff + "[0]", "proj", inject_flag)
                _add_module(f"block{i}_ff_proj2", ff, "2", inject_flag)
            # optionally support time embedding proj if requested like PixArt "emb"
            if "tremb" in target:
                # Time embedding (原来的)
                parent_path_temb = "model.adaln_single.emb.timestep_embedder"
                _add_module("t_embedder_linear1", parent_path_temb, "linear_1", True)
                _add_module("t_embedder_linear2", parent_path_temb, "linear_2", True)
                # Resolution embedding
                parent_path_remb = "model.adaln_single.emb.resolution_embedder" 
                _add_module("r_embedder_linear1", parent_path_remb, "linear_1", True)
                _add_module("r_embedder_linear2", parent_path_remb, "linear_2", True)
                # Aspect ratio embedding
                parent_path_aemb = "model.adaln_single.emb.aspect_ratio_embedder"
                _add_module("a_embedder_linear1", parent_path_aemb, "linear_1", True)
                _add_module("a_embedder_linear2", parent_path_aemb, "linear_2", True)
                # AdaLayerNormSingle中的linear
                parent_path_adaln = "model.adaln_single"
                _add_module("adaln_linear", parent_path_adaln, "linear", True)
            if "cemb" in target:
                # Caption projection
                parent_path_caption = "model.caption_projection"
                _add_module("caption_proj_linear1", parent_path_caption, "linear_1", True)
                _add_module("caption_proj_linear2", parent_path_caption, "linear_2", True)

            # === Step 2: 替换所有模块为 NoisyW8A8Linear (err_prob=0) ===
            all_module_paths = [(p, m) for (p, m, _) in modules_for_select.values()]
            print(f"[PixArt Injection] Replacing {len(all_module_paths)} Linear modules with NoisyW8A8Linear...")

            self.replace_modules_in_parallel(
                target_module_paths=all_module_paths,
                protected=protected,
                weight_quant=weight_quant,
                act_quant=act_quant,
                quantize_bmm_input=quantize_bmm_input,
                err_prob=0.0,        # 默认无错
                err_fn=lambda x: 0.0,   # 默认无错函数
                method=method
            )

            # === Step 3: 为选中的模块生成 err_fn ===
            _, err_fn = parse_target(target, err_prob, modules_for_select, default_modules)

            # === Step 4: 激活目标层的错误注入 ===
            activated = 0
            for key, (parent_path, module_name, should_inject) in modules_for_select.items():
                if should_inject:
                    try:
                        parent = eval(f"self.{parent_path}")
                        mod = getattr(parent, module_name)
                        mod.err_prob = err_prob
                        mod.err_fn = err_fn
                        activated += 1
                    except Exception as e:
                        print(f"[Warning] Failed to activate err_fn for {key}: {e}")
                        raise e

            print(f"[PixArt Injection] Activated error injection for {activated}/{len(all_module_paths)} modules ✅")
            print(self.model)

        elif target.startswith("SD15"):
            """
            SD1.5 UNet fault-injection branch (mimics PixArt branch style)
            - collects all Linear-like modules inside Transformer blocks (attn1/attn2/ff)
            - first replaces them with quantized NoisyW8A8Linear (err_prob=0)
            - then activates err_fn for selected layers
            """

            if not hasattr(self, "model"):
                raise RuntimeError("self.model not found for SD1.5 injection")

            #### 先对整体做无错量化
            modules_for_select = {}
            default_modules = []

            # validate target_layers type (same semantics as PixArt branch)
            if (target_layers is not None and (target_layers != "")) and (not all(isinstance(x, int) for x in target_layers) or not isinstance(target_layers, (list, tuple))):
                raise TypeError("target_layers must be a list or tuple of integers")

            # === count total transformer blocks across down/mid/up to build target set ===
            n_blocks = 0
            for i, block in enumerate(getattr(self.model, "down_blocks", [])):
                atts = getattr(block, "attentions", None)
                if atts is not None:
                    for att in atts:
                        tb = getattr(att, "transformer_blocks", None)
                        if tb is not None:
                            n_blocks += len(tb)
            # mid_block
            mid = getattr(self.model, "mid_block", None)
            if mid is not None:
                atts = getattr(mid, "attentions", None)
                if atts is not None:
                    for att in atts:
                        tb = getattr(att, "transformer_blocks", None)
                        if tb is not None:
                            n_blocks += len(tb)
            # up_blocks
            for i, block in enumerate(getattr(self.model, "up_blocks", [])):
                atts = getattr(block, "attentions", None)
                if atts is not None:
                    for att in atts:
                        tb = getattr(att, "transformer_blocks", None)
                        if tb is not None:
                            n_blocks += len(tb)

            target_set = set(target_layers or range(n_blocks))
            print(f"[SD1.5 Injection] Total transformer blocks: {n_blocks}, Target layers: {sorted(list(target_set))}")

            # === Helper function to register one module ===
            def _add_module(key, parent_path, module_name, should_inject):
                """记录一个待替换模块（parent_path 是相对于 self 的字符串路径，module_name 可为 '0' 表示 ModuleList[0]）"""
                modules_for_select[key] = (parent_path, module_name, should_inject)
                default_modules.append(key)

            # === Step 1: 构造所有候选模块（按 transformer 层全局编号） ===
            global_idx = 0
            # helper to walk block.attentions[*].transformer_blocks[*]
            def _walk_block_attentions(block_prefix, block):
                nonlocal global_idx
                atts = getattr(block, "attentions", None)
                if atts is None:
                    return
                for j, att in enumerate(atts):
                    tblocks = getattr(att, "transformer_blocks", None)
                    if tblocks is None:
                        continue
                    for k in range(len(tblocks)):
                        inject_flag = (global_idx in target_set) and ("full" in target)  # follow PixArt: only full if "full" in target
                        prefix = f"{block_prefix}.attentions[{j}].transformer_blocks[{k}]"

                        # ---- Attention 1 ----
                        attn1 = prefix + ".attn1"
                        _add_module(f"blk{global_idx}_attn1_to_q", f"model.{attn1}", "to_q", inject_flag)
                        _add_module(f"blk{global_idx}_attn1_to_k", f"model.{attn1}", "to_k", inject_flag)
                        _add_module(f"blk{global_idx}_attn1_to_v", f"model.{attn1}", "to_v", inject_flag)
                        _add_module(f"blk{global_idx}_attn1_to_out", f"model.{attn1}.to_out", "0", inject_flag)

                        # ---- Attention 2 ----
                        attn2 = prefix + ".attn2"
                        # attn2 sometimes consumes cross context but attribute name exists in SD1.5 prints -> register same shape
                        _add_module(f"blk{global_idx}_attn2_to_q", f"model.{attn2}", "to_q", inject_flag)
                        _add_module(f"blk{global_idx}_attn2_to_k", f"model.{attn2}", "to_k", inject_flag)
                        _add_module(f"blk{global_idx}_attn2_to_v", f"model.{attn2}", "to_v", inject_flag)
                        _add_module(f"blk{global_idx}_attn2_to_out", f"model.{attn2}.to_out", "0", inject_flag)

                        # ---- FeedForward ----
                        ff = prefix + ".ff.net"
                        _add_module(f"blk{global_idx}_ff_proj1", f"model.{ff}[0]", "proj", inject_flag)
                        _add_module(f"blk{global_idx}_ff_proj2", f"model.{ff}", "2", inject_flag)

                        global_idx += 1

            # Traverse down_blocks
            for i, block in enumerate(getattr(self.model, "down_blocks", [])):
                _walk_block_attentions(f"down_blocks[{i}]", block)

            # mid_block
            if hasattr(self.model, "mid_block"):
                _walk_block_attentions("mid_block", self.model.mid_block)

            # up_blocks
            for i, block in enumerate(getattr(self.model, "up_blocks", [])):
                _walk_block_attentions(f"up_blocks[{i}]", block)

            # optionally support time embedding proj if requested like PixArt "emb"
            if "emb" in target:
                # based on your printed structure: time_embedding.linear_1 / linear_2 inside model.time_embedding
                parent_path_temb = "model.time_embedding"
                # safe attempt: linear_1 & linear_2 are attribute names
                _add_module("time_emb_linear1", parent_path_temb, "linear_1", True)
                _add_module("time_emb_linear2", parent_path_temb, "linear_2", True)

            # === Step 2: 替换所有模块为 NoisyW8A8Linear (err_prob=0) ===
            all_module_paths = [(p, m) for (p, m, _) in modules_for_select.values()]
            print(f"[SD1.5 Injection] Replacing {len(all_module_paths)} Linear modules with NoisyW8A8Linear...")

            self.replace_modules_in_parallel(
                target_module_paths=all_module_paths,
                protected=protected,
                weight_quant=weight_quant,
                act_quant=act_quant,
                quantize_bmm_input=quantize_bmm_input,
                err_prob=0.0,        # 默认无错
                err_fn=lambda x: 0.0,   # 默认无错函数
                method=method
            )

            # === Step 3: 为选中的模块生成 err_fn ===
            _, err_fn = parse_target(target, err_prob, modules_for_select, default_modules)
            print(err_fn(0),err_fn(10))

            # === Step 4: 激活目标层的错误注入 ===
            activated = 0
            for key, (parent_path, module_name, should_inject) in modules_for_select.items():
                # print(modules_for_select.items())
                if should_inject:
                    try:
                        parent = eval(f"self.{parent_path}")
                        mod = getattr(parent, module_name)
                        mod.err_prob = err_prob
                        mod.err_fn = err_fn
                        activated += 1
                    except Exception as e:
                        # keep the same behavior as PixArt branch: print warning and re-raise
                        print(f"[Warning] Failed to activate err_fn for {key}: {e}")
                        raise e

            print(f"[SD1.5 Injection] Activated error injection for {activated}/{len(all_module_paths)} modules ✅")
            print(self.model)

        elif (target == "Skip") or (target == "Debug"):
            print("Do nothing.")
            return None
        
        else:
            raise ValueError(f"不支持的 target: {target}. 检查是否有拼写错误？")

        print(f'Fault injected to {target} finished.')
        return None
    
if __name__ == "__main__":
    # 假设 modules_for_select
    modules_for_select = {
        "block0": ("model.blocks[0].attn", "qkv"),
        "block1": ("model.blocks[1].attn", "proj"),
        "block2": ("model.blocks[2].mlp", "fc1"),
        "block2": ("model.blocks[2].mlp", "fc2"),
    }

    default_modules = ["block0", "block1"]
    # 测试
    target_strs = ["KKK-modules_block0_block1", "KKK-step12t", "KKK-step10t49", "KKK-step0t10-step20t30"]
    err_prob = 0.01

    for target_str in target_strs:
        modules_select, err_fn = parse_target(target_str, err_prob, modules_for_select, default_modules)

        print("modules_select:", modules_select)
        print("Testing err_fn:")

        for step in [0, 5, 10, 15, 18, 20, 25, 30, 45, 50]:
            print(f"Step {step}: err_prob = {err_fn(step)}")

    
