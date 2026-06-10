import json
import random

def load_coco_captions(annotation_path="/data/home/jinqiwen/workspace/diffusion_fault_tolerance/ddim/datasets/coco17/annotations/captions_val2017.json",
                        max_prompts=None,
                        random_sample=False,
                        seed=42):
    """
    从 COCO 2017 标注文件中加载验证集的 captions

    Args:
        annotation_path: captions_val2017.json 的路径
        max_prompts: 最大加载的prompt数量（用于测试，None表示加载全部）
        random_sample: 是否随机采样 max_prompts 条 captions
        seed: 随机种子，保证采样固定

    Returns:
        prompts: list 包含所有caption的列表
        image_ids: list 对应每条prompt的image_id
    """
    print(f"Loading COCO captions from {annotation_path}...")

    with open(annotation_path, 'r') as f:
        data = json.load(f)

    # 创建 image_id 到 captions 的映射
    image_captions = {}
    for ann in data['annotations']:
        image_id = ann['image_id']
        caption = ann['caption']
        if image_id not in image_captions:
            image_captions[image_id] = []
        image_captions[image_id].append(caption)

    # 按 image_id 排序并选择第一个caption
    sorted_image_ids = sorted(image_captions.keys())
    prompts = []
    image_ids = []

    for image_id in sorted_image_ids:
        prompt = image_captions[image_id][0]  # 每张图选择第一个 caption
        prompts.append(prompt)
        image_ids.append(image_id)

    # 如果需要随机采样
    if max_prompts is not None:
        if random_sample:
            random.seed(seed)  # 固定随机种子
            indices = random.sample(range(len(prompts)), min(max_prompts, len(prompts)))
            prompts = [prompts[i] for i in indices]
            image_ids = [image_ids[i] for i in indices]
        else:
            prompts = prompts[:max_prompts]
            image_ids = image_ids[:max_prompts]

    print(f"Loaded {len(prompts)} COCO captions")
    print(f"Example captions:")
    for i in range(min(3, len(prompts))):
        print(f"  {i+1}. {prompts[i]} (image_id={image_ids[i]})")

    return prompts, image_ids
