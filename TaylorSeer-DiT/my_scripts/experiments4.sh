#!/bin/bash
#### todo:
# 一组baseline   !!!!!
# bit 8 12 14 16 20 -1  ******
# layers 0 6 12 18 24 27  ******
# steps 0t2 9t11 19t21 29t31 39t41 47t49  !!!!!
# 单走embedding  !!!!!
# 恢复性（一组带hook即可）
# protect embedding  vs all  !!!!!
# protect first2 vs middle2 vs last2 vs no (step)  %%%%%
# AD22 VS ABFT12+清零 VS ABFT12+cache &&&&&
# ABFT 8 12 14 20  AD 18  (DSE)  &&&&&
# block_size 32 64 128 (DSE)  %%%%%
# cache interval 1 4 10 20  &&&&&

### 6 * 9 * 20min = 18h
#!/bin/bash

export CUDA_VISIBLE_DEVICES=1

# 基础命令
BASE_CMD="python sample.py"
# 参数遍历配置
NUM_INFERENCE_STEPS_LIST=(50)
ERR_PROB_LIST=(1e-5 1e-4 1e-3 1e-2 1e-7 1e-6 3e-4 3e-3 3e-2) 
TARGET_LIST=("DiTXL512full-step0t47" "DiTXL512full-step0t23-step26t49" "DiTXL512full")
TARGET_LAYERS_LIST=(
    ""  # 空表示所有层
)
PROTECT_LIST=("ABFT_12")
CACHE_ORDER_LIST=(0) ###默认零阶保护
BIT_LIST=(-1)            
CACHE_INTERVAL_LIST=(10)   

# 计数器
counter=1

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
                                CMD="$BASE_CMD --num_inference_steps $num_steps --err_prob $err_prob --target $target --protect $protect --cache_order $cache_order --bit $bit --cache_interval $cache_interval"
                                
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

#############################
# 基础命令
BASE_CMD="python sample.py"
# 参数遍历配置
NUM_INFERENCE_STEPS_LIST=(50)
ERR_PROB_LIST=(1e-7 1e-6 1e-5 1e-4 3e-4 1e-3 3e-3 1e-2 3e-2)
TARGET_LIST=("DiTXL512full-step0t47" "DiTXL512full-step0t23-step26t49" "DiTXL512full")
TARGET_LAYERS_LIST=(
    ""  # 空表示所有层
)
PROTECT_LIST=("ABFT_12")
CACHE_ORDER_LIST=(0) ###默认零阶保护
BIT_LIST=(-1)
CACHE_INTERVAL_LIST=(10)
ABFT_BLOCK_SIZE_LIST=(32 64 128)  # 新增参数

# 计数器
counter=1

# 遍历所有参数组合
for num_steps in "${NUM_INFERENCE_STEPS_LIST[@]}"; do
    for err_prob in "${ERR_PROB_LIST[@]}"; do
        for target in "${TARGET_LIST[@]}"; do
            for target_layers in "${TARGET_LAYERS_LIST[@]}"; do
                for protect in "${PROTECT_LIST[@]}"; do
                    for cache_order in "${CACHE_ORDER_LIST[@]}"; do
                        for bit in "${BIT_LIST[@]}"; do
                            for cache_interval in "${CACHE_INTERVAL_LIST[@]}"; do
                                for abft_block_size in "${ABFT_BLOCK_SIZE_LIST[@]}"; do
                                    
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
                                    echo "  abft_block_size: $abft_block_size"
                                    echo "=============================================="
                                    
                                    # 构建命令
                                    CMD="$BASE_CMD --num_inference_steps $num_steps --err_prob $err_prob --target $target --protect $protect --cache_order $cache_order --bit $bit --cache_interval $cache_interval --abft_block_size $abft_block_size"
                                    
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
done

echo "所有实验完成! 总共运行了 $((counter-1)) 个实验"
