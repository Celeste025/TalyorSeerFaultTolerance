#!/bin/bash

# 设置要使用的 GPU
export CUDA_VISIBLE_DEVICES=1  # 修改为你想用的 GPU ID

# 定义参数数组
err_probs=(3e-1 1e-3 1e-2 1e-1 1e-4)
num_inference_steps=(50)
# targets=("Exp_Flux_dev1_layer_double" "Exp_Flux_dev1_layer_dto" "Exp_Flux_dev1_layer_dadd" )
# target_layers=(0 6 12 18)
targets=("Exp_Flux_dev1_layer_double")
target_layers=({0..18..1})

# 定义日志文件路径
log_file="results/run_tests_targets2.log"

# 清空日志文件或创建新的日志文件
> "$log_file"

# 遍历组合
for err_prob in "${err_probs[@]}"; do
    for steps in "${num_inference_steps[@]}"; do
        for target in "${targets[@]}"; do
            for layer in "${target_layers[@]}"; do
                echo "Running with err_prob=${err_prob}, num_inference_steps=${steps}, target=${target}, target_layers=${layer}" | tee -a "$log_file"
                
                python diffusers_taylorseer_flux.py \
                    --err_prob "$err_prob" \
                    --num_inference_steps "$steps" \
                    --target "$target" \
                    --target_layers "$layer" \
                    --max_prompts 20 | tee -a "$log_file"
                
                echo "-------------------------------------------" | tee -a "$log_file"
            done
        done
    done
done

echo "所有测试已完成。结果已保存到 ${log_file}。" | tee -a "$log_file"