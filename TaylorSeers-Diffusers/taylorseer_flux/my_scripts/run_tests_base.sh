
# 设置要使用的 GPU
export CUDA_VISIBLE_DEVICES=0  # 修改为你想用的 GPU ID

# 定义参数数组
err_probs=(0)
num_inference_steps=(30 50)
targets=("Skip")  # 根据你的情况填入

# 遍历组合
for err_prob in "${err_probs[@]}"; do
    for steps in "${num_inference_steps[@]}"; do
        for target in "${targets[@]}"; do
            echo "Running with err_prob=${err_prob}, num_inference_steps=${steps}, target=${target}"
            
            python diffusers_taylorseer_flux.py \
                --err_prob "$err_prob" \
                --num_inference_steps "$steps" \
                --target "$target" \
                --max_prompts 20
            
            echo "-------------------------------------------"
        done
    done
done