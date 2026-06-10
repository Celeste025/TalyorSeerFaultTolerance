#!/bin/bash

# 设置GPU
export CUDA_VISIBLE_DEVICES=0

# 基础命令
BASE_CMD="python sample.py"

# 参数遍历配置
NUM_INFERENCE_STEPS_LIST=(50)
ERR_PROB_LIST=(1e-7 1e-6 1e-5 1e-4 1e-3 3e-3 1e-2)
TARGET_LIST=("DiTXL512full")
TARGET_LAYERS_LIST=(
    ""  # 空表示所有层
)
PROTECT_LIST=("No")
BIT_LIST=(10 12 14 16 18 20)  # 新增：bit 参数列表

# 计数器
counter=1

# 遍历所有参数组合
for num_steps in "${NUM_INFERENCE_STEPS_LIST[@]}"; do
    for err_prob in "${ERR_PROB_LIST[@]}"; do
        for target in "${TARGET_LIST[@]}"; do
            for target_layers in "${TARGET_LAYERS_LIST[@]}"; do
                for protect in "${PROTECT_LIST[@]}"; do
                    for bit in "${BIT_LIST[@]}"; do  # 新增：bit 循环
                        
                        echo "=============================================="
                        echo "运行实验 $counter:"
                        echo "  num_inference_steps: $num_steps"
                        echo "  err_prob: $err_prob"
                        echo "  target: $target"
                        echo "  target_layers: $target_layers"
                        echo "  protect: $protect"
                        echo "  bit: $bit"
                        echo "=============================================="
                        
                        # 构建命令
                        CMD="$BASE_CMD --num_inference_steps $num_steps --err_prob $err_prob --target $target --protect $protect --bit $bit"
                        
                        # 如果target_layers不为空，添加到命令中
                        if [ -n "$target_layers" ]; then
                            CMD="$CMD --target_layers $target_layers"
                        fi
                        
                        # 执行命令
                        echo "执行: $CMD"
                        $CMD
                        
                        # 检查命令是否成功执行
                        if [ $? -eq 0 ]; then
                            echo "实验 $counter 完成 ✅"
                        else
                            echo "实验 $counter 失败 ❌"
                            # exit 1
                        fi
                        
                        echo
                        ((counter++))
                    
                    done
                done
            done
        done
    done
done

echo "所有实验完成! 总共运行了 $((counter-1)) 个实验"
