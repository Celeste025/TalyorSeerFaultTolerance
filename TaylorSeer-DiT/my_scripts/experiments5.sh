#!/bin/bash
#### todo:
# taylorseer 正交性
# interval,order = (3,2) 
# 对于实际推理的step注错，其他step复用cache逻辑即可
#!/bin/bash
# 10 * 50 * 25s = 3.5h
export CUDA_VISIBLE_DEVICES=1

# 基础命令
BASE_CMD="python sample.py"

# 参数遍历配置
NUM_INFERENCE_STEPS_LIST=(50)
ERR_PROB_LIST=(0.0 1e-7 1e-6 1e-5 1e-4 3e-4 1e-3 3e-3 6e-3 1e-2 )
TARGET_LIST=("DiTXL512full-step2t")
TARGET_LAYERS_LIST=(
    ""  # 空表示所有层
)
PROTECT_LIST=("ABFT_12")
CACHE_ORDER_LIST=(0)  # 默认零阶保护
BIT_LIST=(-1)
CACHE_INTERVAL_LIST=(9)  # 是interval的整数倍

# 新增 interval 和 order 的遍历列表
INTERVAL_LIST=(3)
ORDER_LIST=(2)

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
                                for interval in "${INTERVAL_LIST[@]}"; do
                                    for order in "${ORDER_LIST[@]}"; do

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
                                        echo "  interval: $interval"
                                        echo "  order: $order"
                                        echo "=============================================="

                                        # 构建命令
                                        CMD="$BASE_CMD \
                                            --num_inference_steps $num_steps \
                                            --err_prob $err_prob \
                                            --target $target \
                                            --protect $protect \
                                            --cache_order $cache_order \
                                            --bit $bit \
                                            --cache_interval $cache_interval \
                                            --interval $interval \
                                            --max-order $order \
                                            --fig_per_class 10"

                                        # 如果 target_layers 不为空，添加到命令中
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
done

echo "所有实验完成! 总共运行了 $((counter-1)) 个实验"


