#!/bin/bash
# ===========================================
# 遍历多个结果文件夹进行评估并生成 Excel，需在 TaylorSeerFaultTolerance 根目录下运行
# ===========================================

export CUDA_VISIBLE_DEVICES=1

# 可遍历的结果文件夹列表
RESULTS_DIRS=(
    "SD15/results_bench"
)

REF_FOLDER="SD15/results_bench/target_SD15full_step_50_err_prob_0.0_cacheinter_10"

DO_CLIP=true
DO_IMAGE_REWARD=true
DO_LPIPS=true
FORCE_RECOMPUTE=true

# 遍历每个结果文件夹
for RESULTS_DIR in "${RESULTS_DIRS[@]}"; do
    echo "==========================================="
    echo "Processing folder: $RESULTS_DIR"
    echo "==========================================="

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
    echo "Running command:"
    echo "$CMD"
    eval $CMD

    # ===========================================
    # Generate Excel summary and plots
    # ===========================================
    python evaluation/gen_xlsx.py --root_folder "$RESULTS_DIR"
done
