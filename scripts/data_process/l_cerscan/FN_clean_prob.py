import argparse
import json
import os
import re
import shutil
from collections import defaultdict
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from mmengine.fileio import load
from PIL import Image
from pycocotools import mask as mask_utils
from tqdm import tqdm


RUN_DIR = "work_dir/binary_cls/chief/2026_05_19_10_12_08"
PRED_RESULTS = [
    os.path.join(RUN_DIR, "pred_result_train.pkl"),
    os.path.join(RUN_DIR, "pred_result_val.pkl"),
]
DATA_ROOT = "data_resource/LCerScan/WS800"
SAVE_DIR = "work_dir/binary_cls/chief/2026_05_19_10_12_08/FN_clean_prob"
TOPK = 100


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize positive samples with the lowest probabilities from a pred_result.pkl."
    )
    parser.add_argument("--pred-result", nargs="+", default=PRED_RESULTS)
    parser.add_argument("--data-root", default=DATA_ROOT)
    parser.add_argument("--save-dir", default=SAVE_DIR)
    parser.add_argument("--topk", type=int, default=TOPK)
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--clean", action="store_true", help="Remove save-dir before writing results.")
    return parser.parse_args()


def safe_name(name):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def tensor_to_list(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, int):
        return [value]
    return list(value)


def scalar_to_float(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


def rel_img_path(img_path, image_root):
    img_path = os.path.normpath(img_path)
    image_root = os.path.normpath(image_root)
    abs_img = os.path.abspath(img_path)
    abs_root = os.path.abspath(image_root)
    try:
        if os.path.commonpath([abs_img, abs_root]) == abs_root:
            return os.path.relpath(abs_img, abs_root).replace(os.sep, "/")
    except ValueError:
        pass

    marker = f"{image_root.rstrip(os.sep)}{os.sep}"
    if marker in img_path:
        return img_path.split(marker, 1)[1].replace(os.sep, "/")
    marker = f"images{os.sep}"
    if marker in img_path:
        return img_path.split(marker, 1)[1].replace(os.sep, "/")
    return img_path.replace(os.sep, "/")


def decode_mask(segmentation):
    if segmentation is None:
        return None
    decoded = mask_utils.decode(segmentation)
    if decoded.ndim == 3:
        decoded = np.any(decoded, axis=2)
    return decoded.astype(bool)


def load_coco_annotations(coco_path):
    with coco_path.open("r", encoding="utf-8") as f:
        coco = json.load(f)

    cat_info = {}
    for cat in coco["categories"]:
        color = cat.get("color") or [31, 119, 180]
        cat_info[cat["id"]] = {
            "name": cat["name"],
            "color": tuple(v / 255 for v in color),
        }

    image_id_to_file = {img["id"]: img["file_name"] for img in coco["images"]}
    ann_by_file = defaultdict(list)
    for ann in coco["annotations"]:
        file_name = image_id_to_file.get(ann["image_id"])
        if file_name is None:
            continue
        ann = dict(ann)
        cat = cat_info[ann["category_id"]]
        ann["category_name"] = cat["name"]
        ann["color"] = cat["color"]
        ann_by_file[file_name].append(ann)

    for anns in ann_by_file.values():
        anns.sort(key=lambda item: item.get("area", item["bbox"][2] * item["bbox"][3]))
    return ann_by_file


def load_all_coco_annotations(data_root):
    data_root = Path(data_root)
    split_paths = {
        "train": data_root / "annofiles" / "puretrain_cocoformat.json",
        "val": data_root / "annofiles" / "val_cocoformat.json",
    }
    return {
        split: load_coco_annotations(path)
        for split, path in split_paths.items()
    }


def collect_positive_samples(pred_results, image_root, ann_by_split):
    samples = []
    missing_split = 0
    total_results = 0

    for pred_result in pred_results:
        results = load(pred_result)
        total_results += len(results)
        for item in results:
            gt_label = tensor_to_list(item.gt_label)
            if len(gt_label) == 0:
                continue

            rel_path = rel_img_path(item.img_path, image_root)
            split = None
            for candidate in ("train", "val"):
                if rel_path in ann_by_split[candidate]:
                    split = candidate
                    break
            if split is None:
                missing_split += 1
                continue

            samples.append(
                {
                    "split": split,
                    "img_path": item.img_path,
                    "rel_path": rel_path,
                    "gt_label": gt_label,
                    "img_prob": scalar_to_float(item.img_prob),
                }
            )

    samples.sort(key=lambda sample: (sample["img_prob"], sample["split"], sample["rel_path"]))
    return samples, missing_split, total_results


def draw_gt(ax, anns, with_mask=False):
    for ann in anns:
        x, y, w, h = ann["bbox"]
        color = ann["color"]
        if with_mask:
            mask = decode_mask(ann.get("segmentation"))
            if mask is not None and mask.any():
                overlay = np.zeros((*mask.shape, 4), dtype=np.float32)
                overlay[..., :3] = color
                overlay[..., 3] = mask.astype(np.float32) * 0.35
                ax.imshow(overlay)
                ax.contour(mask, levels=[0.5], colors=[color], linewidths=1.0)

        ax.add_patch(
            patches.Rectangle(
                (x, y),
                w,
                h,
                linewidth=1.6,
                edgecolor=color,
                facecolor="none",
            )
        )
        label = f"{ann['category_name']} area={int(ann.get('area', w * h))}"
        ax.text(
            x,
            max(0, y - 4),
            label,
            color="white",
            fontsize=7,
            va="bottom",
            bbox={"facecolor": color, "edgecolor": "none", "alpha": 0.85, "pad": 1},
        )


def save_visual(sample, rank, anns, save_path, dpi):
    image = Image.open(sample["img_path"]).convert("RGB")
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for ax in axes:
        ax.imshow(image)
        ax.axis("off")

    draw_gt(axes[0], anns, with_mask=False)
    axes[0].set_title("GT bbox")
    draw_gt(axes[1], anns, with_mask=True)
    axes[1].set_title("GT bbox + mask")
    title = (
        f"rank={rank} split={sample['split']} prob={sample['img_prob']:.6f} "
        f"gt_num={len(anns)} {sample['rel_path']}"
    )
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    if args.clean and os.path.isdir(args.save_dir):
        shutil.rmtree(args.save_dir)
    os.makedirs(args.save_dir, exist_ok=True, mode=0o777)

    image_root = Path(args.data_root) / "images"
    ann_by_split = load_all_coco_annotations(args.data_root)
    samples, missing_split, total_results = collect_positive_samples(
        args.pred_result,
        image_root,
        ann_by_split,
    )
    selected = samples[: args.topk]

    print(
        f"Loaded {total_results} predictions, collected {len(samples)} positive samples "
        f"with COCO GT annotations."
    )
    if missing_split:
        print(f"Skipped {missing_split} positive samples that were not found in train/val COCO annotations.")
    print(f"Saving lowest-probability top-{len(selected)} samples to {args.save_dir}")

    for rank, sample in enumerate(tqdm(selected, ncols=80, desc="visualize"), start=1):
        anns = ann_by_split[sample["split"]][sample["rel_path"]]
        stem = safe_name(Path(sample["rel_path"]).with_suffix("").as_posix())
        vis_name = f"{rank:03d}_prob{sample['img_prob']:.6f}_{sample['split']}_{stem}.png"
        save_visual(sample, rank, anns, os.path.join(args.save_dir, vis_name), args.dpi)


if __name__ == "__main__":
    main()
