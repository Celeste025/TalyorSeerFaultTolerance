# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Sample new images from a pre-trained DiT.
"""
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torchvision.utils import save_image
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from download import find_model
from models import DiT_models
import argparse

#### Extra imports for fault injection ####
import json
import sys
import os
# 当前脚本所在路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 把 fault_injection 目录加入 sys.path
sys.path.append(os.path.join(BASE_DIR, '..', 'fault_injection'))
# 把 utils 目录加入 sys.path
sys.path.append(os.path.join(BASE_DIR, '..', 'utils'))
# 把 evaluation 目录加入 sys.path
sys.path.append(os.path.join(BASE_DIR, '..', 'evaluation'))
from FIassistant import FIassistant
from InjectionState import _injection_state # 全局单例
from Recorder import _recorder
from HookManager import _hook_manager
from common import * 
from imagenet_helper import ImageNetPromptGenerator


def main(args):
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)  # 如果使用多GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ############ Fault injection setup ############
    additional_str = ""
    if args.analyzer_yes:
        additional_str += "a"
    if args.hook_yes:
        additional_str += "h"
    


    folder = make_result_folder_name(args.target, args.num_inference_steps, args.err_prob, args.target_layers, 
            args.bit, args.protect, args.cache_quant, args.cache_interval, cache_order=args.cache_order, abft_block_size=args.abft_block_size,
            taylorseer_interval=args.interval, taylorseer_max_order=args.max_order, bench=args.bench,
            additional_str=additional_str)
    os.makedirs(folder, exist_ok=True)
    save_run_params(os.path.dirname(folder), vars(args))

    _injection_state.set_inject_bit(args.bit)
    _injection_state.global_args['hook_yes'] = args.hook_yes
    _injection_state.global_args['analyzer_yes'] = args.analyzer_yes
    _injection_state.global_args['folder_path'] = folder
    _injection_state.global_args['protect'] = args.protect
    _injection_state.global_args['cache_quant'] = args.cache_quant
    _injection_state.global_args['cache_interval'] = args.cache_interval // args.interval  # convert to effective interval
    _injection_state.global_args['abft_block_size'] = args.abft_block_size
    print(f"Using protection method: {args.protect}")
    print(f"Using ABFT block size: {args.abft_block_size}")
    if args.cache_order != -1:
        print(f"Using cache order: {args.cache_order}")
        print(f"Using cache quant: {args.cache_quant}")
        print(f"Using cache interval: {args.cache_interval}")
    else:
        print("Not using cache.")
    _injection_state.global_args['cache_order'] = args.cache_order
    _injection_state.global_args['interval'] = args.interval   ####用于支持TaylorSeer

    #########

    if args.ckpt is None:
        assert args.model == "DiT-XL/2", "Only DiT-XL/2 models are available for auto-download."
        assert args.image_size in [256, 512]
        assert args.num_classes == 1000

    # Load model:
    latent_size = args.image_size // 8
    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes
    ).to(device)
    # Auto-download a pre-trained model or load a custom DiT checkpoint from train.py:
    ckpt_path = args.ckpt 
    if ckpt_path is None:
        ckpt_path = f"DiT-XL-2-{args.image_size}x{args.image_size}.pt"
        print(f"Auto-downloading pre-trained DiT-XL/2 model to {ckpt_path} ...")
    state_dict = find_model(ckpt_path)
    model.load_state_dict(state_dict)
    model.eval()  # important!
    diffusion = create_diffusion(str(args.num_inference_steps))
    if args.vae == "ema":
        vae_model_path = "/data/home/jinqiwen/.cache/huggingface/hub/models--stabilityai--sd-vae-ft-ema/snapshots/f04b2c4b98319346dad8c65879f680b1997b204a"
        vae = AutoencoderKL.from_pretrained(vae_model_path, local_files_only=True).to(device)
    elif args.vae == "mse":
        vae_model_path = "/data/home/jinqiwen/.cache/huggingface/hub/models--stabilityai--sd-vae-ft-mse/snapshots/31f26fdeee1355a5c34592e401dd41e45d25a493"
        vae = AutoencoderKL.from_pretrained(vae_model_path, local_files_only=True).to(device)
    else:
        print("No local vae model cache, try downloading from huggingface directly.")
        vae_model_path = "stabilityai/sd-vae-ft-{args.vae}"
    vae = AutoencoderKL.from_pretrained(vae_model_path).to(device)  # can also load from local file

    ##### fault injection setup continued #####
    fiassistant = FIassistant(model)
    fiassistant.inject_fault_to_module(
        target=args.target,
        weight_quant='per_tensor',
        act_quant='per_tensor',
        quantize_bmm_input=True,
        err_prob=args.err_prob,
        target_layers=args.target_layers
    )
    # import pdb;pdb.set_trace()
    ###注册hook
    if args.hook_yes:
        hook_layers = []
        # # 1. 每层的 mlp.fc2 输出
        # for i in range(len(model.blocks)):
        #     hook_layers.append(f"blocks.{i}.mlp.fc2")
        # # 2. 每层的 attn.proj 输出  
        # for i in range(len(model.blocks)):
        #     hook_layers.append(f"blocks.{i}.attn.proj")
        # # 3. 每个 DiTblock 的完整输出（block级别）
        # for i in range(len(model.blocks)):
        #     hook_layers.append(f"blocks.{i}")  # 整个 DiTBlock 的输出
        # # 4. embedding 层
        # hook_layers += [
        #     "x_embedder",                    # 图像patch嵌入
        #     "t_embedder",                    # 时间步嵌入
        #     "y_embedder"                     # 类别标签嵌入
        # ]
        
        # # 5. final_layer 相关（虽然不是严格意义上的embedding，但也很重要）
        # hook_layers += [
        #     "final_layer",                   # 最终输出层
        #     "final_layer.linear",            # 最终线性投影
        #     "final_layer.adaLN_modulation.1" # final_layer的调制线性层
        # ]

        _hook_manager.initialize(model=model, layer_names=hook_layers)
        _hook_manager.register_hooks(capture_mode="output", print_module_names=False)

    # Define class labels to sample and create intial noise:
    if args.bench:
        class_labels = [1, 8, 58, 77, 355, 100, 200, 300, 400, 500]  # goldfish, hen, water snake, wolf spider, llama, black swan, Tibetan terrier, tiger beetle, academic gown, cliff dwelling
        # class_labels = [1, 8, 77, 355, 283, 817, 889, 949, 483, 292]
    else:
        class_labels = [1, 8, 58, 77, 355] # Example class IDs: goldfish, hen, water snake, wolf spider, llama 
    prompts = []
    prompt_generator = ImageNetPromptGenerator()
    for class_id in class_labels:
        p = prompt_generator.get_prompt(class_id)  
        prompts.append(p)    
    print(prompts, class_labels)
    assert len(prompts) == len(class_labels)
    n = len(class_labels)

    for i, prompt in enumerate(prompts):
        print(f"Class ID: {class_labels[i]}, Prompt: {prompt}")
        for j in range(args.fig_per_class):
            generator = torch.Generator(device=device).manual_seed(args.seed + j)
            _injection_state.set_step(0)
            #### start generation ####
            z = torch.randn(1, 4, latent_size, latent_size, device=device, generator=generator)
            y = torch.tensor([class_labels[i]], device=device)
            z = torch.cat([z, z], 0)
            y_null = torch.tensor([1000] * 1, device=device)
            y = torch.cat([y, y_null], 0)
            model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)
            model_kwargs['interval']        = args.interval
            model_kwargs['max_order']       = args.max_order
            model_kwargs['test_FLOPs']      = args.test_FLOPs
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            if args.ddim_sample:
                samples = diffusion.ddim_sample_loop(
                    model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
                )
            else:
                print("fault_injection may not be compatible with p_sample now")
                samples = diffusion.p_sample_loop(
                    model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
                )
            end.record()
            torch.cuda.synchronize()
            print(f"Total Sampling took {start.elapsed_time(end)*0.001} seconds")

            samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
            samples = vae.decode(samples / 0.18215).sample

            # Save images:
            filename_safe_prompt = truncate_filename(prompt.replace(" ", "_"))
            img_name = f"{filename_safe_prompt}_{j}.png"
            txt_name = f"{filename_safe_prompt}_{j}.txt"
            img_path = os.path.join(folder, img_name)
            txt_path = os.path.join(folder, txt_name)
            save_image(samples, img_path, nrow=1, normalize=True, value_range=(-1, 1))
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(prompt)
            print(f"Saved image and prompt to {img_path} and {txt_path}")

            ### 清理缓存：
            fiassistant.clear_all_noisy_linear_caches()
            # import pdb;pdb.set_trace()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="mse")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=512)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=4)
    # parser.add_argument("--num-sampling-steps", type=int, default=50)  ###弃用
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Optional path to a DiT checkpoint (default: auto-download a pre-trained DiT-XL/2 model).")
    parser.add_argument("--ddim-sample", action="store_true", default=True)  # 目前只测试ddim
    parser.add_argument("--interval", type=int, default=1)  ###不应用interval skip
    parser.add_argument("--max-order", type=int, default=0) ###不应用高阶approximation
    parser.add_argument("--test-FLOPs", action="store_true", default=False)

    #### fault_injection related args
    parser.add_argument("--fig_per_class", type=int, default=10)  ###每类默认10张
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--err_prob", type=float, default=0)
    parser.add_argument("--target", type=str, default="DiTXL512full")  
    parser.add_argument("--max_prompts", type=int, default=30)  ###无效参数
    parser.add_argument(
        "--target_layers",
        type=int,
        nargs='+',  # '+' means one or more arguments
        default=[],  # Default list of integers
        help="List of target layers as integers."
    )
    parser.add_argument("--bit", type=int, default=-1)
    parser.add_argument("--analyzer_yes", action='store_true', default=False)
    parser.add_argument("--hook_yes", action='store_true', default=False)
    parser.add_argument("--protect", type=str, default="No")
    parser.add_argument("--cache_order", type=int, default=-1)   #-1表示不使用cache
    parser.add_argument("--cache_quant", type=int, default=8)    #8表示不额外quant了  
    parser.add_argument("--cache_interval", type=int, default=1) #1表示每层都cache
    parser.add_argument("--abft_block_size", type=int, default=32) #仅当protect为ABFT时有效，block size
    parser.add_argument("--bench", action='store_true', default=False)
    args = parser.parse_args()
    if args.interval != 1:
        assert args.cache_interval % args.interval == 0, "cache_interval must be multiple of interval"   
    main(args)
