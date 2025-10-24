import os
import torch
import ImageReward as RM

# 模型单例（懒加载）
_image_reward_model = None
def get_model():
    global _image_reward_model
    if _image_reward_model is None:
        print("Loading ImageReward model ...")
        _image_reward_model = RM.load("ImageReward-v1.0")
    return _image_reward_model

def compute_image_reward_one_image(img_path, prompt=None):
    """
    计算单张图的 ImageReward 分数，自动加载模型
    """
    model = get_model()

    if prompt is None:
        txt_path = os.path.splitext(img_path)[0] + ".txt"
        if os.path.exists(txt_path):
            with open(txt_path, "r", encoding="utf-8") as f:
                prompt = f.read().strip()
        else:
            print(f"Warning: No prompt provided and no txt file found for {img_path}. Using empty prompt.")
            prompt = ""
    
    with torch.no_grad():
        score = model.score(prompt, img_path)
    return score

def compute_image_reward_folder(folder_path, detail=False):
    """
    批量计算文件夹下图片的 ImageReward 分数，自动加载模型
    返回平均分
    """
    scores = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
            img_path = os.path.join(folder_path, filename)
            try:
                score = compute_image_reward_one_image(img_path)
                scores.append(score)
                if detail:
                    print(f"{filename}: {score:.2f}")
            except Exception as e:
                print(f"Skipped {filename}: {e}")

    if not scores:
        print("Warning: No valid images found in folder.")
        return None

    avg_score = sum(scores) / len(scores)
    return avg_score

if __name__ == "__main__":
    # 单张图测试
    single_img_path = "/data/home/jinqiwen/workspace/diffusion_fault_tolerance/TaylorSeerFaultTolerance/evaluation/TaylorSeer_An image of a squirrel in Picasso style.png"
    score = compute_image_reward_one_image(single_img_path, prompt="An image of a squirrel in Picasso style")
    print(f"Reward score for {os.path.basename(single_img_path)}: {score:.2f}")

    # 文件夹批量测试，开启 detail
    folder_path = "/data/home/jinqiwen/workspace/diffusion_fault_tolerance/ddim/results/num_inference_steps_50_err_prob_0_target_UNet_diffusion/images_gen"
    avg_score = compute_image_reward_folder(folder_path, detail=True)
    print(f"Average Reward score for folder {os.path.basename(folder_path)}: {avg_score:.2f}")
    
