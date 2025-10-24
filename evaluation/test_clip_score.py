import os
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import clip
from tqdm import tqdm

# 模型单例（懒加载）
_clip_model = None
_clip_preprocess = None

def get_clip_model(device="cuda"):
    global _clip_model, _clip_preprocess
    if _clip_model is None or _clip_preprocess is None:
        print("Loading CLIP model ...")
        _clip_model, _clip_preprocess = clip.load("ViT-L/14", device=device)
        _clip_model.eval()
    return _clip_model, _clip_preprocess

def compute_clip_score_one_image(img_path, prompt=None, device="cuda"):
    """
    计算单张图像的 CLIP image-text score
    """
    model, preprocess = get_clip_model(device)

    if prompt is None:
        txt_path = os.path.splitext(img_path)[0] + ".txt"
        if os.path.exists(txt_path):
            with open(txt_path, "r", encoding="utf-8") as f:
                prompt = f.read().strip()
        else:
            print(f"Warning: No prompt provided and no txt file found for {img_path}. Using empty prompt.")
            prompt = ""

    img = preprocess(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)
    text_tokens = clip.tokenize([prompt]).to(device)

    with torch.no_grad():
        img_emb = model.encode_image(img)
        text_emb = model.encode_text(text_tokens)

    img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
    text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)

    score = (img_emb * text_emb).sum().item()
    return score

def compute_clip_score_folder(folder_path, detail=False, device="cuda"):
    """
    批量计算文件夹下图片的 CLIP image-text score
    返回平均分
    """
    scores = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
            img_path = os.path.join(folder_path, filename)
            try:
                score = compute_clip_score_one_image(img_path, device=device)
                scores.append(score)
                if detail:
                    print(f"{filename}: {score:.4f}")
            except Exception as e:
                print(f"Skipped {filename}: {e}")

    if not scores:
        print("Warning: No valid images found in folder.")
        return 0.0

    avg_score = sum(scores) / len(scores)
    return avg_score

if __name__ == "__main__":
    # 示例：单张图测试
    single_img_path = "/data/home/jinqiwen/workspace/diffusion_fault_tolerance/TaylorSeerFaultTolerance/evaluation/TaylorSeer_An image of a squirrel in Picasso style.png"
    prompt = "An image of a squirrel in Picasso style"
    score = compute_clip_score_one_image(single_img_path, prompt)
    print(f"CLIP score for {os.path.basename(single_img_path)}: {score:.4f}")

    # 示例：文件夹批量测试
    folder_path = "/data/home/jinqiwen/workspace/diffusion_fault_tolerance/ddim/results/num_inference_steps_50_err_prob_0_target_UNet_diffusion/images_gen"
    avg_score = compute_clip_score_folder(folder_path, detail=True)
    print(f"Average CLIP score for folder {os.path.basename(folder_path)}: {avg_score:.4f}")