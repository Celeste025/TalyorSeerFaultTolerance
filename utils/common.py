import json
import sys
import os
import re
from typing import Any, Dict, Optional, Tuple, Union
def sanitize_filename(s: str, max_length: int = 50) -> str:
    """
    将任意字符串转换成文件名安全的形式：
    - 替换所有非字母数字的字符为下划线
    - 限制长度，防止过长
    """
    sanitized = re.sub(r'[^a-zA-Z0-9_\-]', '_', s)
    return sanitized[:max_length]

def load_coco_captions(annotation_path="/data/home/jinqiwen/workspace/diffusion_fault_tolerance/ddim/datasets/coco17/annotations/captions_val2017.json",
                        max_prompts=None):
    """
    从 COCO 2017 标注文件中加载验证集的 captions
    
    Args:
        annotation_path: captions_val2017.json 的路径
        max_prompts: 最大加载的prompt数量（用于测试，None表示加载全部）
    
    Returns:
        prompts: list 包含所有caption的列表，按image_id排序
        image_ids: list 对应每条prompt的image_id
    """
    print(f"Loading COCO captions from {annotation_path}...")
    
    with open(annotation_path, 'r') as f:
        data = json.load(f)
    
    # 创建 image_id 到 captions 的映射
    image_captions = {}
    for ann in data['annotations']:
        image_id = ann['image_id']
        caption = ann['caption']
        if image_id not in image_captions:
            image_captions[image_id] = []
        image_captions[image_id].append(caption)
    
    # 按 image_id 排序并选择第一个caption（标准做法）
    sorted_image_ids = sorted(image_captions.keys())
    prompts = []
    image_ids = []
    
    for image_id in sorted_image_ids:
        prompt = image_captions[image_id][0]
        prompts.append(prompt)
        image_ids.append(image_id)
        
        if max_prompts and len(prompts) >= max_prompts:
            break
    
    print(f"Loaded {len(prompts)} COCO captions")
    print(f"Example captions:")
    for i in range(min(3, len(prompts))):
        print(f"  {i+1}. {prompts[i]} (image_id={image_ids[i]})")
    
    return prompts, image_ids

def make_result_folder_name(target: str, num_inference_steps: int, err_prob: float, target_layers, bit=-1, protect="No", 
    cache_quant=8, cache_interval=1, cache_order=0, abft_block_size=32, 
    taylorseer_interval=1, taylorseer_max_order=0, bench=False, additional_str=None) -> str:
    folder_name = f"target_{target}_step_{num_inference_steps}_err_prob_{err_prob}"
    if target_layers:  #[] or list of int
        folder_name += "_layers"
        for i in target_layers:
            folder_name += "_"
            folder_name += str(i)
    if bit != -1:
        folder_name += f"_bit_{bit}"
    if protect != "No":
        folder_name += f"_protect_{protect}"
    
    if additional_str:
        folder_name += f"_{additional_str}"
    if bench: #生成数较多的测试组，高优先级
        base_dir = "results_bench2"
    elif taylorseer_interval != 1:
        base_dir = "results_taylorseer"
    elif target_layers:
        base_dir = "results_layers"
    elif protect != "No":
        base_dir = "results_protect"
    else:
        base_dir = "results"
    
    if cache_quant != 8:
        folder_name += f"_cachequant_{cache_quant}"
    if cache_interval != 1:
        folder_name += f"_cacheinter_{cache_interval}"
    if protect.startswith("ABFT") and cache_order != 0:   ###补丁，一般而言我们的ABFT方法需要cache支持
        folder_name += f"_cacheorder_{cache_order}"
    if protect.startswith("ABFT") and abft_block_size != 32:
        folder_name += f"_abftblock_{abft_block_size}"

    if taylorseer_interval != 1:  #TaylorSeer启动
        folder_name += f"_tinter_{taylorseer_interval}"
        folder_name += f"_torder_{taylorseer_max_order}"


    return os.path.join(base_dir, folder_name, "images_gen")

def save_run_params(folder: str, args: dict):
    os.makedirs(folder, exist_ok=True)
    json_path = os.path.join(folder, "run_params.json")
    if not os.path.exists(json_path):
        with open(json_path, "w") as f:
            json.dump(args, f, indent=2)

def truncate_filename(s: str, max_len: int = 40):
    return s if len(s) <= max_len else s[:max_len]