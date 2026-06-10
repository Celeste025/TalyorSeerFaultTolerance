#!/usr/bin/env python3
"""
ImageNet subset extraction + FID evaluation pipeline using clean-fid.
- real_output_dir and gen_folder provided via CLI arguments
- Extracts real images by matching user-provided label ids (0-999) using WNID mapping
"""

import shutil
import tarfile
import json
import argparse
from pathlib import Path
from cleanfid import fid

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
IMAGENET_ROOT = Path("evaluation/datasets/imagenet")
VAL_TAR_FILE = IMAGENET_ROOT / "ILSVRC2012_img_val.tar"
VAL_EXTRACTED_DIR = IMAGENET_ROOT / "val"
IMAGENET_LABEL_MAP = IMAGENET_ROOT / "imagenet_class_index.json"
VAL_GT_FILE = IMAGENET_ROOT / "ILSVRC2012_validation_ground_truth.txt"  # 50000 labels

# ---------------------------------------------------------------------------
# Download / Extract
# ---------------------------------------------------------------------------
def download_imagenet_if_needed():
    if VAL_TAR_FILE.exists():
        print("[Info] Found existing ImageNet val tar.")
        return
    print("[Error] ImageNet validation tar not found:", VAL_TAR_FILE)
    print("Please manually download ILSVRC2012_img_val.tar (6.3GB)")
    raise SystemExit

def extract_val_if_needed():
    if VAL_EXTRACTED_DIR.exists():
        print("[Info] ImageNet val already extracted.")
        return
    print("[Info] Extracting ImageNet val...")
    VAL_EXTRACTED_DIR.mkdir(parents=True, exist_ok=True)
    with tarfile.open(VAL_TAR_FILE) as tar:
        tar.extractall(path=VAL_EXTRACTED_DIR)
    print("[Info] Extraction complete.")

# ---------------------------------------------------------------------------
# Load label mapping
# ---------------------------------------------------------------------------
def load_label_map():
    with open(IMAGENET_LABEL_MAP, 'r') as f:
        idx2wnid = json.load(f)
    # label_id (0-999) -> WNID
    return {int(k): v[0] for k, v in idx2wnid.items()}

# ---------------------------------------------------------------------------
# Load ground-truth validation labels
# ---------------------------------------------------------------------------
def load_val_gt():
    with open(VAL_GT_FILE, 'r') as f:
        labels = [int(x.strip()) for x in f.readlines()]
        print(max(labels), min(labels))
    if len(labels) != 50000:
        raise ValueError(f"Expected 50000 labels, got {len(labels)}")
    return labels

# ---------------------------------------------------------------------------
# Collect real images corresponding to label_ids
# ---------------------------------------------------------------------------
def collect_real_images(label_ids, output_dir):
    # 清空已有内容
    if output_dir.exists():
        print(f"[Info] Clearing existing folder: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[Info] Loading label map...")
    idx2wnid = load_label_map()         # 0-999 -> WNID
    wnid_to_label = {v: k for k, v in idx2wnid.items()}

    print("[Info] Loading validation ground-truth labels...")
    gt_labels = load_val_gt()           # 50000 labels
    selected_files = []

    for i, label in enumerate(gt_labels):
        if label in label_ids:
            img_name = f"ILSVRC2012_val_{i+1:08d}.JPEG"
            src = VAL_EXTRACTED_DIR / img_name
            dst = output_dir / img_name
            if src.exists():
                shutil.copy(src, dst)
                selected_files.append(dst)

    print(f"[Info] Collected {len(selected_files)} real images → {output_dir}")
    # import pdb; pdb.set_trace()

# ---------------------------------------------------------------------------
# Compare counts
# ---------------------------------------------------------------------------
def compare_counts(real_dir, gen_dir):
    n_real = len(list(real_dir.glob("*.JPEG"))) + len(list(real_dir.glob("*.jpg")))
    n_gen = len(list(gen_dir.glob("*.png"))) + len(list(gen_dir.glob("*.jpg"))) + len(list(gen_dir.glob("*.jpeg")))
    print(f"Real: {n_real}, Gen: {n_gen}")
    if n_real != n_gen:
        print("[Warning] Count mismatch!")
    else:
        print("[OK] Numbers match.")

# -----------------------------------------------------------------a----------
# Compute FID
# ---------------------------------------------------------------------------
def compute_fid(real_dir, gen_dir):
    print("[Info] Computing FID...")
    score = fid.compute_fid(str(real_dir), str(gen_dir), mode="clean", num_workers=32)
    print(f"FID = {score}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real_output", type=str,
                        default="TaylorSeer-DiT/results_bench/target_DiTXL512full-step2t_step_50_err_prob_0.003_protect_ABFT_10_cacheinter_10/tmp")
    parser.add_argument("--gen_folder", type=str,
                        default="TaylorSeer-DiT/results_bench/target_DiTXL512full-step2t_step_50_err_prob_0.003_protect_ABFT_10_cacheinter_10/images_gen")
    parser.add_argument("--labels", nargs="*", type=int,
                        default=[158, 186, 384, 421, 450, 483, 609, 621, 713, 896])
                        # default=[158])
    args = parser.parse_args()
    # print("label += 1 for matching ImageNet labels.")
    # args.labels = [l for l in args.labels]  # Adjust for 1-based indexing in GT file
    real_dir = Path(args.real_output)
    gen_dir = Path(args.gen_folder)
    label_ids = args.labels

    print("Using label IDs:", label_ids)

    download_imagenet_if_needed()
    extract_val_if_needed()
    collect_real_images(label_ids, real_dir)
    compare_counts(real_dir, gen_dir)
    compute_fid(real_dir, gen_dir)

if __name__ == "__main__":
    main()


# [Matched] target='Tibetan terrier' -> ID=158, WNID=n02097474, words='Tibetan terrier, chrysanthemum dog', gloss='breed of medium-sized terriers bred in Tibet resembling Old English sheepdogs with fluffy curled tails'
# [Matched] target='llama' -> ID=186, WNID=n02437616, words='llama', gloss='wild or domesticated South American cud-chewing animal related to camels but smaller and lacking a hump'
# [Matched] target='hen' -> ID=384, WNID=n01514859, words='hen', gloss='adult female bird'
# [Matched] target='black swan' -> ID=421, WNID=n01860187, words='black swan, Cygnus atratus', gloss='large Australian swan having black plumage and a red bill'
# [Matched] target='goldfish' -> ID=450, WNID=n01443537, words='goldfish, Carassius auratus', gloss='small golden or orange-red freshwater fishes of Eurasia used as pond or aquarium fishes'
# [Matched] target='water snake' -> ID=483, WNID=n01737021, words='water snake', gloss='any of various mostly harmless snakes that live in or near water'
# [Matched] target='wolf spider' -> ID=609, WNID=n01775062, words='wolf spider, hunting spider', gloss='ground spider that hunts its prey instead of using a web'
# [Matched] target='tiger beetle' -> ID=621, WNID=n02165105, words='tiger beetle', gloss='active usually bright-colored beetle that preys on other insects'
# [Matched] target='cliff dwelling' -> ID=713, WNID=n03042490, words='cliff dwelling', gloss='a rock and adobe dwelling built on sheltered ledges in the sides of a cliff; "the Anasazi built cliff dwellings in the southwestern United States"'
# [Matched] target='academic gown' -> ID=896, WNID=n02669723, words='academic gown, academic robe, judge's robe', gloss='a gown worn by academics or judges'

# Final mapping:
# Tibetan terrier: 158
# llama: 186
# hen: 384
# black swan: 421
# goldfish: 450
# water snake: 483
# wolf spider: 609
# tiger beetle: 621
# cliff dwelling: 713
# academic gown: 896

# import scipy.io
# from pathlib import Path

# def find_ids_for_classes(meta_path, target_classes):
#     """
#     meta_path: path to meta.mat
#     target_classes: list of keywords, e.g., ["goldfish", "hen", "wolf spider"]
    
#     Returns:
#         dict: target_class -> ILSVRC2012_ID
#     """
#     meta_path = Path(meta_path)
#     meta = scipy.io.loadmat(str(meta_path))
#     synsets = meta['synsets']  # shape (1860, 1)

#     class_to_id = {}

#     for i in range(synsets.shape[0]):
#         s = synsets[i, 0]  # 注意这里是 i,0
#         class_id = int(s['ILSVRC2012_ID'][0][0])
#         wnid = s['WNID'][0]
#         words = s['words'][0]  # 不转换小写，先打印原始
#         gloss = s['gloss'][0]
#         words_list = [w.strip().lower() for w in words.split(',')]  # 小写，拆分
#         for target in target_classes:
#             target_lower = target.lower()
#             if any(target_lower == w for w in words_list):  # 完全匹配
#                 class_to_id[target] = class_id
#                 print(f"[Matched] target='{target}' -> ID={class_id}, WNID={wnid}, words='{words}', gloss='{gloss}'")

#     return class_to_id

# # 调用示例
# meta_file = "evaluation/datasets/imagenet/meta.mat"
# targets = ["goldfish", "hen", "water snake", "wolf spider", "llama", 
#            "black swan", "Tibetan terrier", "tiger beetle", "academic gown", "cliff dwelling"]

# mapping = find_ids_for_classes(meta_file, targets)
# print("\nFinal mapping:")
# for k, v in mapping.items():
#     print(f"{k}: {v}")
