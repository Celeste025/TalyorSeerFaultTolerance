
export CUDA_VISIBLE_DEVICES=5
# 计数器
counter=1

#################################
#baseline
# 基础命令
BASE_CMD="python sample.py"

# # 参数遍历配置
# NUM_INFERENCE_STEPS_LIST=(30)
# ERR_PROB_LIST=(0 1e-5 1e-4 1e-3 1e-2 1e-7 1e-6 3e-4 3e-3 3e-2 1e-8)  
# TARGET_LIST=("SD15full")
# TARGET_LAYERS_LIST=(
#     ""  # 空表示所有层
# )
# PROTECT_LIST=("No")
# CACHE_ORDER_LIST=(-1)
# BIT_LIST=(-1)            
# CACHE_INTERVAL_LIST=(1)   



# # 遍历所有参数组合
# for num_steps in "${NUM_INFERENCE_STEPS_LIST[@]}"; do
#     for err_prob in "${ERR_PROB_LIST[@]}"; do
#         for target in "${TARGET_LIST[@]}"; do
#             for target_layers in "${TARGET_LAYERS_LIST[@]}"; do
#                 for protect in "${PROTECT_LIST[@]}"; do
#                     for cache_order in "${CACHE_ORDER_LIST[@]}"; do
#                         for bit in "${BIT_LIST[@]}"; do
#                             for cache_interval in "${CACHE_INTERVAL_LIST[@]}"; do
                                
#                                 echo "=============================================="
#                                 echo "运行实验 $counter:"
#                                 echo "  num_inference_steps: $num_steps"
#                                 echo "  err_prob: $err_prob"
#                                 echo "  target: $target"
#                                 echo "  target_layers: $target_layers"
#                                 echo "  protect: $protect"
#                                 echo "  cache_order: $cache_order"
#                                 echo "  bit: $bit"
#                                 echo "  cache_interval: $cache_interval"
#                                 echo "=============================================="
                                
#                                 # 构建命令
#                                 CMD="$BASE_CMD --num_inference_steps $num_steps --err_prob $err_prob --target $target --protect $protect --cache_order $cache_order --bit $bit --cache_interval $cache_interval --max_prompts 5"
                                
#                                 # 如果target_layers不为空，添加到命令中
#                                 if [ -n "$target_layers" ]; then
#                                     CMD="$CMD --target_layers $target_layers"
#                                 fi
                                
#                                 # 执行命令
#                                 echo "执行: $CMD"
#                                 $CMD
                                
#                                 # 检查命令是否成功执行
#                                 if [ $? -eq 0 ]; then
#                                     echo "实验 $counter 完成 ✅"
#                                 else
#                                     echo "实验 $counter 失败 ❌"
#                                     # exit 1
#                                 fi
                                
#                                 echo
#                                 ((counter++))
                                
#                             done
#                         done
#                     done
#                 done
#             done
#         done
#     done
# done

# echo "所有实验完成! 总共运行了 $((counter-1)) 个实验"


# #################################
# # bit
# # 基础命令
# BASE_CMD="python sample.py"

# # 参数遍历配置
# NUM_INFERENCE_STEPS_LIST=(50)
# ERR_PROB_LIST=(1e-5 1e-4 1e-3 1e-2 1e-7 1e-6 3e-4 3e-3 3e-2)  
# TARGET_LIST=("SD15full")
# TARGET_LAYERS_LIST=(
#     ""  # 空表示所有层
# )
# PROTECT_LIST=("No")
# CACHE_ORDER_LIST=(-1)
# BIT_LIST=(8 10 12 14 16)            
# CACHE_INTERVAL_LIST=(1)   



# # 遍历所有参数组合
# for num_steps in "${NUM_INFERENCE_STEPS_LIST[@]}"; do
#     for err_prob in "${ERR_PROB_LIST[@]}"; do
#         for target in "${TARGET_LIST[@]}"; do
#             for target_layers in "${TARGET_LAYERS_LIST[@]}"; do
#                 for protect in "${PROTECT_LIST[@]}"; do
#                     for cache_order in "${CACHE_ORDER_LIST[@]}"; do
#                         for bit in "${BIT_LIST[@]}"; do
#                             for cache_interval in "${CACHE_INTERVAL_LIST[@]}"; do
                                
#                                 echo "=============================================="
#                                 echo "运行实验 $counter:"
#                                 echo "  num_inference_steps: $num_steps"
#                                 echo "  err_prob: $err_prob"
#                                 echo "  target: $target"
#                                 echo "  target_layers: $target_layers"
#                                 echo "  protect: $protect"
#                                 echo "  cache_order: $cache_order"
#                                 echo "  bit: $bit"
#                                 echo "  cache_interval: $cache_interval"
#                                 echo "=============================================="
                                
#                                 # 构建命令
#                                 CMD="$BASE_CMD --num_inference_steps $num_steps --err_prob $err_prob --target $target --protect $protect --cache_order $cache_order --bit $bit --cache_interval $cache_interval --max_prompts 5"
                                
#                                 # 如果target_layers不为空，添加到命令中
#                                 if [ -n "$target_layers" ]; then
#                                     CMD="$CMD --target_layers $target_layers"
#                                 fi
                                
#                                 # 执行命令
#                                 echo "执行: $CMD"
#                                 $CMD
                                
#                                 # 检查命令是否成功执行
#                                 if [ $? -eq 0 ]; then
#                                     echo "实验 $counter 完成 ✅"
#                                 else
#                                     echo "实验 $counter 失败 ❌"
#                                     # exit 1
#                                 fi
                                
#                                 echo
#                                 ((counter++))
                                
#                             done
#                         done
#                     done
#                 done
#             done
#         done
#     done
# done

# echo "所有实验完成! 总共运行了 $((counter-1)) 个实验"

##################################
###保护一下试试
# 参数遍历配置
NUM_INFERENCE_STEPS_LIST=(30)
ERR_PROB_LIST=(1e-5 1e-4 1e-3 1e-2 1e-7 1e-6 3e-4 3e-3 3e-2 1e-8)  
TARGET_LIST=("SD15full-step1t")
TARGET_LAYERS_LIST=(
    ""  # 空表示所有层
)
PROTECT_LIST=("ABFT_12")
CACHE_ORDER_LIST=(0)
BIT_LIST=(-1)            
CACHE_INTERVAL_LIST=(10)   



# 遍历所有参数组合
for num_steps in "${NUM_INFERENCE_STEPS_LIST[@]}"; do
    for err_prob in "${ERR_PROB_LIST[@]}"; do
        for target in "${TARGET_LIST[@]}"; do
            for target_layers in "${TARGET_LAYERS_LIST[@]}"; do
                for protect in "${PROTECT_LIST[@]}"; do
                    for cache_order in "${CACHE_ORDER_LIST[@]}"; do
                        for bit in "${BIT_LIST[@]}"; do
                            for cache_interval in "${CACHE_INTERVAL_LIST[@]}"; do
                                
                                echo "=============================================="
                                echo "运行实验 $counter:"
                                echo "  num_inference_steps: $num_steps"
                                echo "  err_prob: $err_prob"
                                echo "  target: $target"
                                echo "  target_layers: $target_layers"
                                echo "  protect: $protect"
                                echo "  cache_order: $cache_order"
                                echo "  bit: $bit"
                                echo "  cache_interval: $cache_interval"
                                echo "=============================================="
                                
                                # 构建命令
                                CMD="$BASE_CMD --num_inference_steps $num_steps --err_prob $err_prob --target $target --protect $protect --cache_order $cache_order --bit $bit --cache_interval $cache_interval --max_prompts 5"
                                
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
    done
done

echo "所有实验完成! 总共运行了 $((counter-1)) 个实验"
