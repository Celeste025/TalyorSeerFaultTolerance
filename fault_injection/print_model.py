import torch
import os

def print_model(model: torch.nn.Module, name: str = "dummy"):
    """
    打印模型结构并输出到文件
    
    参数:
        model: PyTorch模型
        name: 用于生成输出文件名的标识符
    """
    
    # 1. 输出完整模型结构
    struct_file = f"model_struct_{name}.txt"
    with open(struct_file, "w") as f:
        # 使用str(model)获取完整结构
        f.write(f"Full structure of model '{name}':\n")
        f.write(str(model))
        f.write("\n\nParameters count:\n")
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        f.write(f"Total params: {total_params:,}\n")
        f.write(f"Trainable params: {trainable_params:,}\n")
        f.write(f"Non-trainable params: {total_params - trainable_params:,}\n")
    
    # 2. 输出所有层名称
    name_file = f"model_name_{name}.txt"
    with open(name_file, "w") as f:
        f.write(f"Layer names of model '{name}':\n")
        for module_name, module in model.named_modules():
            # 跳过顶层模块（模型本身）
            if module_name == "":
                continue
            f.write(f"{module_name} - {type(module).__name__}\n")
    
    print(f"Model structure saved to {struct_file}")
    print(f"Layer names saved to {name_file}")


def extract_linear_layers(model: torch.nn.Module, model_name = "dummy"):
    """
    提取模型中所有Linear层的名称并保存到文件
    
    参数:
        model: PyTorch模型
        model_name: 用于生成输出文件名的标识符
    """
    # 输出文件路径
    output_file = f"linear_layers_{model_name}.txt"
    
    # 收集所有Linear层
    linear_layers = []
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Linear, torch.nn.modules.linear.NonDynamicallyQuantizableLinear)):
            # 获取完整的层路径
            full_name = name
            # 添加层详细信息
            layer_info = f"{full_name} - in_features={module.in_features}, out_features={module.out_features}, bias={module.bias is not None}"
            linear_layers.append(layer_info)
    
    # 写入文件
    with open(output_file, "w") as f:
        f.write(f"Linear layers in model '{model_name}':\n")
        f.write("="*50 + "\n")
        for layer in linear_layers:
            f.write(layer + "\n")
        
        # 添加统计信息
        f.write("\n" + "="*50 + "\n")
        f.write(f"Total Linear layers: {len(linear_layers)}\n")
    
    print(f"Linear layers info saved to {output_file}")
    return linear_layers
