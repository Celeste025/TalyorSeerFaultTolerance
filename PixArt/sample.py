import torch
from diffusers import PixArtAlphaPipeline, ConsistencyDecoderVAE, AutoencoderKL
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
from coco_helper import load_coco_captions
from drawbench_helper import DrawBenchPromptGenerator

def main(args):
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)  # 如果使用多GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    print(f"[Before] Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")

    #### 设置fault injection相关参数
    if (args.analyzer_yes or args.hook_yes) and (args.max_prompts != 1):
        print("Warning. It is strongly recommended to set max_prompts to 1 when using analyzer_yes and hook_yes, here we will only run the last prompt.")
    additional_str = ""
    if args.analyzer_yes:
        additional_str += "a"
    if args.hook_yes:
        additional_str += "h"
    
    folder = make_result_folder_name(args.target, args.num_inference_steps, args.err_prob, args.target_layers, 
            args.bit, args.protect, args.cache_quant, args.cache_interval, cache_order=args.cache_order, abft_block_size=args.abft_block_size,
            bench=args.bench, additional_str=additional_str)
    os.makedirs(folder, exist_ok=True)
    save_run_params(os.path.dirname(folder), vars(args))

    _injection_state.set_inject_bit(args.bit)
    _injection_state.global_args['hook_yes'] = args.hook_yes
    _injection_state.global_args['analyzer_yes'] = args.analyzer_yes
    _injection_state.global_args['folder_path'] = folder
    _injection_state.global_args['protect'] = args.protect
    _injection_state.global_args['cache_quant'] = args.cache_quant
    _injection_state.global_args['cache_interval'] = args.cache_interval
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
    #########


    #### 加载pipeline
    #pipe = PixArtAlphaPipeline.from_pretrained("PixArt-alpha/PixArt-XL-2-1024-MS", torch_dtype=torch.float16, use_safetensors=True)
    local_model_path = "/data/home/jinqiwen/.cache/huggingface/hub/models--PixArt-alpha--PixArt-XL-2-1024-MS/snapshots/b89adadeccd9ead2adcb9fa2825d3fabec48d404"
    pipe = PixArtAlphaPipeline.from_pretrained(local_model_path, torch_dtype=torch.float16, use_safetensors=True)
    # print(pipe.scheduler) 用的DPM-Solver
    pipe.to(device)

    ##### fault injection setup continued #####
    model = pipe.transformer
    fiassistant = FIassistant(model)
    fiassistant.inject_fault_to_module(
        target=args.target,
        weight_quant='per_channel',
        act_quant='per_token',
        quantize_bmm_input=True,
        err_prob=args.err_prob,
        target_layers=args.target_layers
    )
    # import pdb;pdb.set_trace()
    ###注册hook, 待补充
    if args.hook_yes:
        hook_layers = []
        _hook_manager.initialize(model=model, layer_names=hook_layers)
        _hook_manager.register_hooks(capture_mode="output", print_module_names=False)
    
    ###生成prompts
    if args.hook_yes:
        prompts = ["a cactus standing in a desert, digital art"]  #测试用固定prompt
    else:
        # prompts, image_ids = load_coco_captions(max_prompts=args.max_prompts)
        generator = DrawBenchPromptGenerator()
        idxs_prompts = generator.sample_prompts()
        prompts = [prompt for idx, prompt in idxs_prompts[:args.max_prompts]]
        print(len(prompts), "prompts loaded.")
    
    for prompt in prompts:
        for j in range(args.fig_per_prompt):
            torch.manual_seed(args.seed + j)
            torch.cuda.manual_seed_all(args.seed + j)  # 如果使用多GPU
            _injection_state.set_step(0)
            def my_callback(step: int, timestep: int, latents: torch.Tensor):
                if args.hook_yes:
                    save_dir = os.path.join(_injection_state.global_args['folder_path'], "layer_out")
                    os.makedirs(save_dir, exist_ok=True)
                    save_name = f"step_latent_{step}.pt"
                    torch.save(latents.cpu(), os.path.join(save_dir, save_name))
                    print("Saved latents to ", os.path.join(save_dir, save_name))
                    # ===== 解码成 image =====
                    with torch.no_grad():
                        # 每个模型 decode latent 方式不同
                        # PixArtAlphaPipeline 使用 dec=pipe.vae.decode()
                        imgs = pipe.vae.decode(latents / pipe.vae_scale_factor).sample  # [1, 3, H, W]
                        # save image
                        img = pipe.image_processor.postprocess(imgs, output_type="pil")[0]
                        img.save(os.path.join(save_dir, f"image_step_{step}.png"))
                        print("Saved image to ", os.path.join(save_dir, f"image_step_{step}.png"))

                ###########
                _injection_state.set_step(step + 1) # 更新当前step，供fault injection使用, 回调函数是step结束后调用的，所以应该set step+1
                # print("Step:", _injection_state.current_step())
                
            image = pipe(prompt, num_inference_steps=30, callback=my_callback, callback_steps=1).images[0]
            # Save images:
            filename_safe_prompt = truncate_filename(prompt.replace(" ", "_"))
            img_name = f"{filename_safe_prompt}_{j}.png"
            txt_name = f"{filename_safe_prompt}_{j}.txt"
            img_path = os.path.join(folder, img_name)
            txt_path = os.path.join(folder, txt_name)
            image.save(img_path)
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(prompt)
            print(f"Saved image and prompt to {img_path} and {txt_path}")

            ### 清理缓存：
            fiassistant.clear_all_noisy_linear_caches()

    # 推理后显存
    print(f"[After] Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    print(f"[Peak] Max allocated: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")

if __name__ == "__main__":
    #### fault_injection related args
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fig_per_prompt", type=int, default=1)
    parser.add_argument("--num_inference_steps", type=int, default=30)
    parser.add_argument("--err_prob", type=float, default=0.0)
    parser.add_argument("--target", type=str, default="Skip")  
    parser.add_argument("--max_prompts", type=int, default=50)  
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
    main(args)