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

    return modules_select(list of tuples)
    """
    step_range = None  # (start, end) 元组，end=None 表示无上限
    modules = []

    # 解析 step
    step_match = re.search(r'-step(\d+(?:t\d*)?)', target)  # 匹配 12, 5t15, 5t
    if step_match:
        step_str = step_match.group(1)
        if 't' in step_str:
            start_str, end_str = step_str.split('t')
            start = int(start_str)
            end = int(end_str) if end_str else None  # None 表示无上限
            step_range = (start, end)
        else:
            step_range = (int(step_str), int(step_str))  # 单步也用元组表示

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
    if step_range is not None:
        start, end = step_range
        def err_fn(current_step):
            if end is None:
                return err_prob if current_step >= start else 0.0
            else:
                return err_prob if start <= current_step <= end else 0.0
    else:
        # 未指定 step，则 step >= 0 返回 err_prob，step < 0 返回 0
        def err_fn(current_step):
            return err_prob if current_step >= 0 else 0.0
    
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
                    print(f"[Error] 并行构造模块失败: {e}")

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

        elif target.startswith("UNet_diffusion"):  # eg ""UNet_diffusion-step12-modules_d0t_u1t"

            # 每个 down/up/mid block 抽取一个linear层
            modules_for_select = {
                # ---- Down blocks (down 0..3) ----
                "d0o2" :('model.down_blocks[0].attentions[-1].transformer_blocks[-1].attn2.to_out', '0'),
                "d1o2" :('model.down_blocks[1].attentions[-1].transformer_blocks[-1].attn2.to_out', '0'),  #attn2是cross-attn
                # 这几个层仅用于实验
                "d1o1" :('model.down_blocks[1].attentions[-1].transformer_blocks[-1].attn1.to_out', '0'),  #attn1是self-attn
                "d1ff" :('model.down_blocks[1].attentions[-1].transformer_blocks[-1].ff.net', '2'),        #跟在attn1,attn2后面的feed-forward层
                "d1t"  :('model.down_blocks[1].resnets[-1]', 'time_emb_proj'),                             #resnet中的time_emb_proj
                ####
                "d2o2" :('model.down_blocks[2].attentions[-1].transformer_blocks[-1].attn2.to_out', '0'),

                # down_blocks[3] 没有 attentions -> 使用最后一个 resnet 的 time_emb_proj
                "d3t" :('model.down_blocks[3].resnets[-1]', 'time_emb_proj'),

                # ---- Mid block ----
                "m0o" :('model.mid_block.attentions[-1].transformer_blocks[-1].attn2.to_out', '0'),

                # ---- Up blocks (up 0..3) ----
                # up_blocks[0] 没有 attentions -> 使用最后一个 resnet 的 time_emb_proj
                "u0t" :('model.up_blocks[0].resnets[-1]', 'time_emb_proj'),
                "u1o2" :('model.up_blocks[1].attentions[-1].transformer_blocks[-1].attn2.to_out', '0'),
                "u2o2" :('model.up_blocks[2].attentions[-1].transformer_blocks[-1].attn2.to_out', '0'),
                "u3o2" :('model.up_blocks[3].attentions[-1].transformer_blocks[-1].attn2.to_out', '0')
            }
            default_modules = ["d0o2","d1o2","d2o2","d3t","m0o","u0t","u1o2","u2o2","u3o2"]
            target_module_paths, err_fn = parse_target(target, err_prob, modules_for_select, default_modules)


            if target_module_paths == -1:  # 表示选择所有模块
                # 遍历模型里所有 Linear
                for name, module in self.model.named_modules():
                    if isinstance(module, nn.Linear):
                        parts = name.split('.')
                        parent_path = '.'.join(parts[:-1])
                        module_name = parts[-1]

                        parent = resolve_module(self.model, parent_path) if parent_path else self.model

                        new_module = create_noisy_module(
                            module, protected, weight_quant, act_quant,
                            quantize_bmm_input, err_prob, method, err_fn
                        )

                        if re.fullmatch(r'-?\d+', module_name):
                            parent[int(module_name)] = new_module
                        else:
                            setattr(parent, module_name, new_module)
            else:
                for parent_path, module_name in target_module_paths:
                    parent = eval(f'self.{parent_path}')
                    original_module = getattr(parent, module_name)

                    if not protected:
                        new_module = NoisyW8A8Linear.from_float(
                            original_module,
                            weight_quant=weight_quant,
                            act_quant=act_quant,
                            quantize_output=quantize_bmm_input,
                            err_prob=err_prob,
                            err_fn=err_fn
                        )
                    else:
                        new_module = NoisyW8A8LinearProtected.from_float(
                            original_module,
                            weight_quant=weight_quant,
                            act_quant=act_quant,
                            quantize_output=quantize_bmm_input,
                            err_prob=err_prob,
                            method=method,
                            err_fn=err_fn
                        )

                    setattr(parent, module_name, new_module)

        elif target == "DiT-XL512":
            print("Do nothing.")
            return None
        elif target.startswith("Flux_dev1"):
            if not hasattr(self, "model"): 
                raise RuntimeError("self.model not found for FLUX-dev1 injection") 
            modules_for_select = {} 
            default_modules = [] 
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



        elif target == "Skip":
            print("Do nothing.")
            return None
        
        else:
            raise ValueError(f"不支持的 target: {target}. 检查是否有拼写错误？")

        print(f'Fault injected to {target} finished.')
        return None

    
