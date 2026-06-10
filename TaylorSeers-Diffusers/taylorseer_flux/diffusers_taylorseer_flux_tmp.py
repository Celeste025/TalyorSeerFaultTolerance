from typing import Any, Dict, Optional, Tuple, Union
import time
from diffusers import DiffusionPipeline
from diffusers.pipelines.flux import FluxPipeline
from diffusers.models import FluxTransformer2DModel
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.utils import USE_PEFT_BACKEND, is_torch_version, logging, scale_lora_layers, unscale_lora_layers
import torch
import numpy as np
from forwards import (taylorseer_flux_single_block_forward, 
                        taylorseer_flux_double_block_forward, 
                        taylorseer_flux_forward)
import argparse
import json
import sys
import os
# 当前脚本所在路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 把 fault_injection 目录加入 sys.path
sys.path.append(os.path.join(BASE_DIR, '..', '..', 'fault_injection'))
# 把 utils 目录加入 sys.path
sys.path.append(os.path.join(BASE_DIR, '..', '..', 'utils'))
from common import *
from FIassistant import FIassistant
from Recorder import _recorder
from HookManager import HookManager


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_inference_steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--err_prob", type=float, default=1e-3)
    parser.add_argument("--target", type=str, default="Flux_dev1")
    parser.add_argument("--max_prompts", type=int, default=20)
    parser.add_argument(
        "--target_layers",
        type=int,
        nargs='+',  # '+' means one or more arguments
        default=[],  # Default list of integers
        help="List of target layers as integers."
    )
    parser.add_argument("--analyzer_yes", action='store_true', default=False)
    parser.add_argument("--hook_yes", action='store_true', default=False)
    args = parser.parse_args()

    if (args.analyzer_yes or args.hook_yes) and (args.max_prompts != 1):
        raise ValueError("It is strongly recommended to set max_prompts to 1 when using analyzer_yes and hook_yes")

    folder = make_result_folder_name(args.target, args.num_inference_steps, args.err_prob)
    os.makedirs(folder, exist_ok=True)
    save_run_params(os.path.dirname(folder), vars(args))

    prompts, image_ids = load_coco_captions(max_prompts =  args.max_prompts)
    print(prompts)

    pipeline = DiffusionPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.float16)
    pipeline.transformer.__class__.num_steps = args.num_inference_steps
    pipeline.transformer.__class__.forward = taylorseer_flux_forward

    for double_block in pipeline.transformer.transformer_blocks:
        double_block.__class__.forward = taylorseer_flux_double_block_forward
    for single_block in pipeline.transformer.single_transformer_blocks:
        single_block.__class__.forward = taylorseer_flux_single_block_forward

    pipeline.to("cuda")
    fiassistant = FIassistant(pipeline.transformer)
    fiassistant.inject_fault_to_module(
        target=args.target,
        weight_quant='per_channel',
        act_quant='per_token',
        quantize_bmm_input=True,
        err_prob=args.err_prob,
        target_layers=args.target_layers
    )

    hook_layers = []
    model = pipeline.transformer
    # transformer_blocks
    for i, block in enumerate(model.transformer_blocks):
        hook_layers.append(f"transformer_blocks.{i}")  # 整个 block 输出
        # hook attention 内部
        # hook_layers.append(f"transformer_blocks.{i}.attn.to_out.0")   # to_out Linear 层
        # hook_layers.append(f"transformer_blocks.{i}.attn.to_add_out")  # add_out Linear 层

    # single_transformer_blocks
    for i, block in enumerate(model.single_transformer_blocks):
        hook_layers.append(f"single_transformer_blocks.{i}")

    # embedding 层
    hook_layers += ["x_embedder",
        "context_embedder", 
        "pos_embed", 
        "time_text_embed",     # time_text_embed 内部 embedder
        "time_text_embed.time_proj",
        "time_text_embed.timestep_embedder",
        "time_text_embed.guidance_embedder",
        "time_text_embed.text_embedder"]

    hook_manager = HookManager(model, layer_names=hook_layers)
    hook_manager.register_hooks(capture_mode="both", print_module_names=False)


    for prompt, img_id in zip(prompts, image_ids):
        generator = torch.Generator("cpu").manual_seed(args.seed)

        def make_hook_callback(hook_manager, hook_yes, save_dir="hook_results"):
            def hook_callback(step, timestep, latents):
                if not hook_yes or hook_manager is None:
                    return
                hook_save_dir = os.path.join(save_dir, "layer_inout1")
                os.makedirs(hook_save_dir, exist_ok=True)
                hook_save_name = f"step_{step}.pt"
                hook_manager.save_outputs(save_dir=hook_save_dir, save_name=hook_save_name)
            return hook_callback

        callback_fn = make_hook_callback(hook_manager, args.hook_yes)

        img = pipeline(
            prompt,
            num_inference_steps=args.num_inference_steps,
            generator=generator,
            callback=callback_fn,
            callback_steps=1
        ).images[0]

        filename_safe_prompt = truncate_filename(prompt.replace(" ", "_"))
        img_name = f"{img_id}_{filename_safe_prompt}.png"
        txt_name = f"{img_id}_{filename_safe_prompt}.txt"

        img.save(os.path.join(folder, img_name))
        with open(os.path.join(folder, txt_name), "w", encoding="utf-8") as f:
            f.write(prompt)

        print(f"Saved image and prompt for ID {img_id}")


if __name__ == "__main__":
    main()
# logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

# num_inference_steps = 30
# seed = 42
# prompt = "A serene sunset over a calm lake, with mountains in the background and soft clouds, realistic style"
# pipeline = DiffusionPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.float16)
# #pipeline = DiffusionPipeline.from_pretrained("/root/autodl-tmp/black-forest-labs/FLUX.1-dev", torch_dtype=torch.float16)
# #pipeline.enable_model_cpu_offload() #save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power

# # TaylorSeer settings
# pipeline.transformer.__class__.num_steps = num_inference_steps

# pipeline.transformer.__class__.forward = taylorseer_flux_forward

# for double_transformer_block in pipeline.transformer.transformer_blocks:
#     double_transformer_block.__class__.forward = taylorseer_flux_double_block_forward
    
# for single_transformer_block in pipeline.transformer.single_transformer_blocks:
#     single_transformer_block.__class__.forward = taylorseer_flux_single_block_forward

# pipeline.to("cuda")
# fiassistant = FIassistant(pipeline.transformer)
# fiassistant.inject_fault_to_module(target="Flux_dev1", weight_quant='per_channel', act_quant='per_token', quantize_bmm_input=True, err_prob=1e-3, target_layers=[])
# parameter_peak_memory = torch.cuda.max_memory_allocated(device="cuda")
# torch.cuda.reset_peak_memory_stats()
# #start_time = time.time()
# start = torch.cuda.Event(enable_timing=True)
# end = torch.cuda.Event(enable_timing=True)
# start.record()
# img = pipeline(
#     prompt, 
#     num_inference_steps=num_inference_steps,
#     generator=torch.Generator("cpu").manual_seed(seed)
#     ).images[0]
# end.record()
# torch.cuda.synchronize()
# elapsed_time = start.elapsed_time(end) * 1e-3
# peak_memory = torch.cuda.max_memory_allocated(device="cuda")

# img.save("{}.png".format('TaylorSeer_' + prompt))

# print(
#     f"epoch time: {elapsed_time:.2f} sec, parameter memory: {parameter_peak_memory/1e9:.2f} GB, memory: {peak_memory/1e9:.2f} GB"
# )