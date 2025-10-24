import re

def parse_target(target: str, err_prob: float, modules_for_select: dict, default_modules: list = None):
    """
    解析文件名安全的 target 字符串，例如：
        UNet_diffusion-step12-modules_d0t_u1t

    参数：
        target: str, 文件名安全的 target
        modules_for_select: dict, {short_name: full_path}
        default_modules: list of short_name, 如果 modules 不存在则使用此列表
        err_prob: float, 错误注入概率
    返回：
        modules: list of short_name
        err_fn: lambda(step) -> err_prob
    """
    step = None
    modules = []

    # 解析 step
    step_match = re.search(r'-step(\d+)', target)
    if step_match:
        step = int(step_match.group(1))

    # 解析 modules
    modules_match = re.search(r'-modules_([A-Za-z0-9_]+)', target)
    if modules_match:
        modules = modules_match.group(1).split('_')
        # 校验 modules 是否在 modules_for_select
        invalid = [m for m in modules if m not in modules_for_select]
        if invalid:
            raise ValueError(f"target 中包含非法模块名: {invalid}")
    elif default_modules is not None:
        # 使用默认模块列表
        invalid = [m for m in default_modules if m not in modules_for_select]
        if invalid:
            raise ValueError(f"default_modules 中包含非法模块名: {invalid}")
        modules = default_modules
    else:
        # 默认全部模块
        modules = list(modules_for_select.keys())

    # 构造 err_fn
    if step is not None:
        # 指定 step 时，只在该 step 返回 err_prob
        def err_fn(current_step):
            return err_prob if current_step == step else 0.0
    else:
        # 未指定 step，则所有 step 都返回 err_prob
        def err_fn(current_step):
            return err_prob

    modules_select = [modules_for_select[m] for m in modules]
    return modules_select, err_fn

if __name__ == "__main__":
    target = "UNet_diffusion"
    err_prob = 0.1
    modules_for_select = {
    "d0t": ('model.action_model.net.history_embedder', 'linear'),
    "u1t": ('model.action_model.net.history_embedder', '0'),
    "m0t": ('model.action_model.net.history_embedder', 'linear2')
    }
    default_modules = ["m0t"]
    modules_select, err_fn = parse_target(target, err_prob, modules_for_select, default_modules)
    print(modules_select)
    print(err_fn(0), err_fn(1), err_fn(2), err_fn(12))