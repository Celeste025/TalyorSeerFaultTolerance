import os
import sys
import json
import argparse

# 确保父文件夹加入 Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from test_image_reward import compute_image_reward_one_image, compute_image_reward_folder
from test_clip_score import compute_clip_score_one_image, compute_clip_score_folder
from test_lpips_score import compute_lpips_score_one_image, compute_lpips_score_folder

### JSON 保存
def update_metrics_json(folder_path, metrics_dict):
    json_path = os.path.join(folder_path, "run_params.json")
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            params = json.load(f)
    else:
        params = {}
    params.update(metrics_dict)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(params, f, indent=4, ensure_ascii=False)

def load_existing_metrics(folder_path):
    json_path = os.path.join(folder_path, "run_params.json")
    if not os.path.exists(json_path):
        return {}
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

### 单图评估
def evaluate_single_image(img_path, prompt=None, ref_folder=None, do_clip=True, do_image_reward=True, do_lpips=True, device="cuda"):
    print(f"\n=== 单张图评估: {img_path} ===")
    results_dict = {}

    if do_clip:
        score = compute_clip_score_one_image(img_path, prompt)
        print(f"CLIP score: {score:.4f}")
        results_dict["clip_score"] = score

    if do_image_reward:
        score = compute_image_reward_one_image(img_path, prompt)
        print(f"ImageReward score: {score:.4f}")
        results_dict["image_reward_score"] = score

    if do_lpips:
        if ref_folder is None:
            print("⚠️ LPIPS 需要提供参考图像文件夹，跳过")
        else:
            score = compute_lpips_score_one_image(img_path, ref_folder)
            print(f"LPIPS score: {score:.4f}")
            results_dict["lpips_score"] = score

    # 保存到同名 json
    folder_path = os.path.dirname(img_path)
    update_metrics_json(folder_path, results_dict)

### 单文件夹评估
def evaluate_folder(images_dir, ref_folder=None, do_clip=True, do_image_reward=True, do_lpips=True, detail=False, force_recompute=False):
    if not os.path.exists(images_dir) or not os.listdir(images_dir):
        print(f"⚠️ 跳过空目录: {images_dir}")
        return

    print(f"\n=== 评估目录: {images_dir} ===")
    results_dict = {}
    parent_folder = os.path.dirname(images_dir)
    existing_metrics = load_existing_metrics(parent_folder)

    if do_clip:
        if not force_recompute and "clip_score" in existing_metrics:
            print(f"跳过 CLIP score（已有结果: {existing_metrics['clip_score']:.4f}）")
        else:
            avg_score = compute_clip_score_folder(images_dir, detail=detail)
            print(f"平均 CLIP score: {avg_score:.4f}")
            results_dict["clip_score"] = avg_score

    if do_image_reward:
        if not force_recompute and "image_reward_score" in existing_metrics:
            print(f"跳过 ImageReward（已有结果: {existing_metrics['image_reward_score']:.4f}）")
        else:
            avg_score = compute_image_reward_folder(images_dir, detail=detail)
            if avg_score:
                print(f"平均 ImageReward score: {avg_score:.4f}")
                results_dict["image_reward_score"] = avg_score

    if do_lpips:
        if ref_folder is None:
            print("LPIPS 需要提供参考图像文件夹，跳过")
        else:
            if not force_recompute and "lpips_score" in existing_metrics:
                print(f"跳过 LPIPS（已有结果: {existing_metrics['lpips_score']:.4f}）")
            else:
                avg_score = compute_lpips_score_folder(images_dir, ref_folder, detail=detail)
                print(f"平均 LPIPS score: {avg_score:.4f}")
                results_dict["lpips_score"] = avg_score

    if results_dict:
        parent_folder = os.path.dirname(images_dir)
        update_metrics_json(parent_folder, results_dict)
        print(f"✅ 已写入 run_params.json")
    else:
        print("✅ 所有指标已存在，跳过重算")

### 多文件夹评估
def evaluate_multi_folder(results_dir, ref_folder=None, do_clip=True, do_image_reward=True, do_lpips=True, detail=False, force_recompute=False):
    if not os.path.exists(results_dir):
        print(f"❌ results_dir 不存在 {results_dir}")
        return
    if ref_folder is not None:
        ref_folder = os.path.join(ref_folder,"images_gen")
        if not os.path.exists(ref_folder):
            print(f"❌ LPIPS 参考文件夹不存在或无images_gen: {ref_folder}")
            return

    for subfolder in os.listdir(results_dir):
        folder_path = os.path.join(results_dir, subfolder)
        if os.path.isdir(folder_path):
            images_dir = os.path.join(folder_path, "images_gen")
            evaluate_folder(
                images_dir,
                ref_folder=ref_folder,
                do_clip=do_clip,
                do_image_reward=do_image_reward,
                do_lpips=do_lpips,
                detail=detail,
                force_recompute=force_recompute
            )
    print("\n✅ 全部子文件夹评估完成！")

def main():
    parser = argparse.ArgumentParser(description="统一图像评估器（封装单图、单文件夹、多文件夹）")
    parser.add_argument("--image_path", type=str, default=None, help="单张图路径")
    parser.add_argument("--folder_path", type=str, default=None, help="单文件夹路径")
    parser.add_argument("--results_dir", type=str, default=None, help="多文件夹处理总目录")
    parser.add_argument("--multi_folder", action="store_true", help="是否遍历 results_dir 下所有子文件夹")
    parser.add_argument("--ref_folder", type=str, default=None, help="LPIPS 参考文件夹")
    parser.add_argument("--do_clip", action="store_true", help="计算 CLIP score")
    parser.add_argument("--do_image_reward", action="store_true", help="计算 ImageReward score")
    parser.add_argument("--do_lpips", action="store_true", help="计算 LPIPS score")
    parser.add_argument("--detail", action="store_true", help="打印每张图分数")
    parser.add_argument("--force_recompute", action="store_true", help="强制重新计算，即使已有结果")
    parser.add_argument("--prompt", type=str, default=None, help="单张图的 prompt")
    args = parser.parse_args()

    if args.multi_folder:
        if args.results_dir is None:
            print("❌ multi_folder=True 时必须提供 --results_dir")
            return
        evaluate_multi_folder(
            args.results_dir,
            ref_folder=args.ref_folder,
            do_clip=args.do_clip,
            do_image_reward=args.do_image_reward,
            do_lpips=args.do_lpips,
            detail=args.detail,
            force_recompute=args.force_recompute
        )
    elif args.image_path:
        evaluate_single_image(
            args.image_path,
            prompt=args.prompt,
            ref_folder=args.ref_folder,
            do_clip=args.do_clip,
            do_image_reward=args.do_image_reward,
            do_lpips=args.do_lpips
        )
    elif args.folder_path:
        evaluate_folder(
            args.folder_path,
            ref_folder=args.ref_folder,
            do_clip=args.do_clip,
            do_image_reward=args.do_image_reward,
            do_lpips=args.do_lpips,
            detail=args.detail,
            force_recompute=args.force_recompute
        )
    else:
        print("❌ 必须提供 --image_path 或 --folder_path 或 --results_dir")

if __name__ == "__main__":
    main()
