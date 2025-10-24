#!/bin/bash

# 设置要使用的 GPU
export CUDA_VISIBLE_DEVICES=2  # 修改为你想用的 GPU ID

# 定义参数数组
err_probs=(1e-3 1e-2 1e-1 3e-1 1e-4)
num_inference_steps=(50)
targets=("Flux_dev1-step0t1" "Flux_dev1-step6t10" "Flux_dev1-step18t22" "Flux_dev1-step30t34" "Flux_dev1-step42t46")
#均覆盖两个有效step
# 定义日志文件路径
log_file="results/run_tests_err_steps.log"

# 清空日志文件或创建新的日志文件
> "$log_file"

# 遍历组合
for err_prob in "${err_probs[@]}"; do
    for steps in "${num_inference_steps[@]}"; do
        for target in "${targets[@]}"; do
            # 打印实际执行的命令
            echo "Executing command: python diffusers_taylorseer_flux.py --err_prob $err_prob --num_inference_steps $steps --target $target --max_prompts 20" | tee -a "$log_file"
            
            python diffusers_taylorseer_flux.py \
                --err_prob "$err_prob" \
                --num_inference_steps "$steps" \
                --target "$target" \
                --max_prompts 20 | tee -a "$log_file"
            
            echo "-------------------------------------------" | tee -a "$log_file"
        done
    done
done

echo "所有测试已完成。结果已保存到 ${log_file}。" | tee -a "$log_file"
