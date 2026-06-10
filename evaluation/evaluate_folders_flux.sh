#!/bin/bash
# ===========================================
# 用于遍历 results_dir 下所有子文件夹进行评估, 请在TaylorSeerFaultTolerance根目录下运行
# ===========================================

export CUDA_VISIBLE_DEVICES=0

RESULTS_DIR="TaylorSeers-Diffusers/taylorseer_flux/results_layers2"
REF_FOLDER="TaylorSeers-Diffusers/taylorseer_flux/results/target_Skip_step_50_err_prob_0.0"

DO_CLIP=true
DO_IMAGE_REWARD=true
DO_LPIPS=true
FORCE_RECOMPUTE=false

# ===========================================
# 构造 Python 命令
# ===========================================
CMD="python evaluation/test_matrics.py --multi_folder --results_dir \"$RESULTS_DIR\""

if [ -n "$REF_FOLDER" ]; then
    CMD+=" --ref_folder \"$REF_FOLDER\""
fi

if [ "$DO_CLIP" = true ]; then
    CMD+=" --do_clip"
fi

if [ "$DO_IMAGE_REWARD" = true ]; then
    CMD+=" --do_image_reward"
fi

if [ "$DO_LPIPS" = true ]; then
    CMD+=" --do_lpips"
fi

if [ "$FORCE_RECOMPUTE" = true ]; then
    CMD+=" --force_recompute"
fi

# 打印并执行
echo "==========================================="
echo "Running command:"
echo "$CMD"
echo "==========================================="
eval $CMD
