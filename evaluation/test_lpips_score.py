import os
import torch
import lpips
import matplotlib.pyplot as plt
import numpy as np

# 模型单例（懒加载）
_lpips_model = None

def get_lpips_model(net='vgg', use_gpu=True):
    global _lpips_model
    if _lpips_model is None:
        print(f"Loading LPIPS model with backbone '{net}' ...")
        _lpips_model = lpips.LPIPS(net=net, spatial=True)
        if use_gpu:
            _lpips_model = _lpips_model.cuda()
    return _lpips_model

def compute_lpips_score_one_image(img_path1, img_path2, net='vgg', use_gpu=True, show_heatmap=False, save_dir=None):
    """
    计算两张图像的 LPIPS score
    """
    model = get_lpips_model(net, use_gpu)

    img0 = lpips.im2tensor(lpips.load_image(img_path1))
    img1 = lpips.im2tensor(lpips.load_image(img_path2))
    if use_gpu:
        img0, img1 = img0.cuda(), img1.cuda()

    with torch.no_grad():
        dist_map = model(img0, img1)  # (1,1,H,W)
        dist_score = dist_map.mean().item()
    
    heatmap = dist_map[0,0].detach().cpu().numpy()
    heatmap_norm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

    if show_heatmap and save_dir:
        plt.figure(figsize=(8,3))
        plt.subplot(1,3,1)
        plt.imshow(lpips.load_image(img_path1))
        plt.title("Ref Image")
        plt.axis("off")
        plt.subplot(1,3,2)
        plt.imshow(lpips.load_image(img_path2))
        plt.title("Gen Image")
        plt.axis("off")
        plt.subplot(1,3,3)
        plt.imshow(heatmap_norm, cmap='jet')
        plt.colorbar()
        plt.title(f"LPIPS Map\n(score={dist_score:.4f})")
        plt.axis("off")
        plt.savefig(save_dir)
        print(f"Heatmap saved to {save_dir}")

    return dist_score, heatmap_norm

def compute_lpips_score_folder(ref_folder, gen_folder, net='vgg', use_gpu=True, detail=False, show_heatmap=False, save_heatmap_dir=None):
    """
    批量计算生成文件夹与参考文件夹中同名图片的 LPIPS score
    返回平均 score
    """
    scores = []
    for filename in os.listdir(gen_folder):
        if filename.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
            gen_path = os.path.join(gen_folder, filename)
            ref_path = os.path.join(ref_folder, filename)
            if not os.path.exists(ref_path):
                print(f"Skipped {filename}: reference image not found in {ref_folder}")
                continue
            save_dir = None
            if show_heatmap and save_heatmap_dir:
                os.makedirs(save_heatmap_dir, exist_ok=True)
                save_dir = os.path.join(save_heatmap_dir, f"{filename}_lpips.png")

            try:
                score, _ = compute_lpips_score_one_image(gen_path, ref_path, net=net, use_gpu=use_gpu,
                                                show_heatmap=show_heatmap, save_dir=save_dir)
                scores.append(score)
                if detail:
                    print(f"{filename}: {score:.4f}")
            except Exception as e:
                print(f"Skipped {filename}: {e}")

    if not scores:
        print("Warning: No valid images found for comparison.")
        return 0.0

    avg_score = sum(scores) / len(scores)
    return avg_score

if __name__ == "__main__":
    # 示例：批量对比生成图像与参考图像（同名匹配）
    ref_folder = "/data/home/jinqiwen/workspace/diffusion_fault_tolerance/ddim/results/num_inference_steps_30_err_prob_0_target_UNet_diffusion/images_gen"
    gen_folder = "/data/home/jinqiwen/workspace/diffusion_fault_tolerance/ddim/results/num_inference_steps_30_err_prob_1e-3_target_UNet_diffusion/images_gen"
    avg_score = compute_lpips_score_folder(ref_folder, gen_folder, detail=True, show_heatmap=False, save_heatmap_dir="./heatmaps")
    print(f"Average LPIPS score: {avg_score:.4f}")