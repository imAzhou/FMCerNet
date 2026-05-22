import argparse
import json
import os
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils
from tqdm import tqdm


LCERSCAN_ROOT = "data_resource/LCerScan"
SAVE_DIR = "work_dir/data_process/FN_clean"
AREA_THR = 200


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize LCerScan GT instances whose annotation.area is smaller than a threshold."
    )
    parser.add_argument("--data-root", default=LCERSCAN_ROOT)
    parser.add_argument("--save-dir", default=SAVE_DIR)
    parser.add_argument("--area-thr", type=float, default=AREA_THR)
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--clean", action="store_true", help="Remove save-dir before writing results.")
    return parser.parse_args()


def safe_name(name):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def decode_mask(segmentation):
    if segmentation is None:
        return None
    decoded = mask_utils.decode(segmentation)
    if decoded.ndim == 3:
        decoded = np.any(decoded, axis=2)
    return decoded.astype(bool)


def split_name(coco_path):
    if coco_path.name.startswith("puretrain"):
        return "puretrain"
    if coco_path.name.startswith("val"):
        return "val"
    return coco_path.stem


def find_coco_files(data_root):
    return sorted(Path(data_root).glob("WS*/annofiles/*_cocoformat.json"))


def load_coco_items(coco_path, area_thr):
    with coco_path.open("r", encoding="utf-8") as f:
        coco = json.load(f)

    cat_info = {}
    for cat in coco["categories"]:
        color = cat.get("color") or [31, 119, 180]
        cat_info[cat["id"]] = {
            "name": cat["name"],
            "color": tuple(v / 255 for v in color),
        }

    image_by_id = {img["id"]: img for img in coco["images"]}
    anns_by_image = defaultdict(list)
    small_ann_ids_by_image = defaultdict(list)

    for ann in coco["annotations"]:
        image = image_by_id.get(ann["image_id"])
        if image is None:
            continue
        ann = dict(ann)
        cat = cat_info[ann["category_id"]]
        ann["category_name"] = cat["name"]
        ann["color"] = cat["color"]
        ann["is_small"] = float(ann.get("area", 0)) < area_thr
        anns_by_image[ann["image_id"]].append(ann)
        if ann["is_small"]:
            small_ann_ids_by_image[ann["image_id"]].append(ann["id"])

    items = []
    for image_id, small_ann_ids in small_ann_ids_by_image.items():
        image = image_by_id[image_id]
        anns = sorted(
            anns_by_image[image_id],
            key=lambda item: (not item["is_small"], item.get("area", item["bbox"][2] * item["bbox"][3])),
        )
        small_anns = [ann for ann in anns if ann["is_small"]]
        items.append(
            {
                "image": image,
                "anns": anns,
                "small_anns": small_anns,
                "small_ann_ids": small_ann_ids,
                "min_area": min(float(ann.get("area", 0)) for ann in small_anns),
            }
        )

    items.sort(key=lambda item: (item["min_area"], item["image"]["file_name"]))
    return items


def draw_boxes(ax, anns, with_mask=False):
    for ann in anns:
        x, y, w, h = ann["bbox"]
        color = ann["color"]
        edgecolor = "yellow" if ann["is_small"] else color
        linewidth = 3.0 if ann["is_small"] else 1.3
        alpha = 0.55 if ann["is_small"] else 0.28

        if with_mask:
            mask = decode_mask(ann.get("segmentation"))
            if mask is not None and mask.any():
                overlay = np.zeros((*mask.shape, 4), dtype=np.float32)
                overlay[..., :3] = color
                overlay[..., 3] = mask.astype(np.float32) * alpha
                ax.imshow(overlay)
                ax.contour(mask, levels=[0.5], colors=[edgecolor], linewidths=linewidth)

        ax.add_patch(
            patches.Rectangle(
                (x, y),
                w,
                h,
                linewidth=linewidth,
                edgecolor=edgecolor,
                facecolor="none",
            )
        )
        prefix = "SMALL " if ann["is_small"] else ""
        label = f"{prefix}{ann['category_name']} area={int(ann.get('area', w * h))}"
        ax.text(
            x,
            max(0, y - 4),
            label,
            color="black" if ann["is_small"] else "white",
            fontsize=7,
            va="bottom",
            bbox={
                "facecolor": "yellow" if ann["is_small"] else color,
                "edgecolor": "none",
                "alpha": 0.9,
                "pad": 1,
            },
        )


def save_visual(image_path, item, title, save_path, dpi):
    image = Image.open(image_path).convert("RGB")
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for ax in axes:
        ax.imshow(image)
        ax.axis("off")

    draw_boxes(axes[0], item["anns"], with_mask=False)
    axes[0].set_title("GT bbox")
    draw_boxes(axes[1], item["anns"], with_mask=True)
    axes[1].set_title("GT bbox + mask")
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    if args.clean and os.path.isdir(args.save_dir):
        shutil.rmtree(args.save_dir)
    os.makedirs(args.save_dir, exist_ok=True, mode=0o777)

    coco_files = find_coco_files(args.data_root)
    if not coco_files:
        raise FileNotFoundError(f"No *_cocoformat.json files found under {args.data_root}")

    count_by_ws = Counter()
    image_count_by_ws = Counter()

    for coco_path in coco_files:
        ws = coco_path.parents[1].name
        split = split_name(coco_path)
        image_root = coco_path.parents[1] / "images"
        save_subdir = Path(args.save_dir) / ws / split
        save_subdir.mkdir(parents=True, exist_ok=True)

        items = load_coco_items(coco_path, args.area_thr)
        image_count_by_ws[ws] += len(items)
        count_by_ws[ws] += sum(len(item["small_anns"]) for item in items)
        print(f"{ws}/{split}: {sum(len(item['small_anns']) for item in items)} small GTs in {len(items)} images")

        for idx, item in enumerate(tqdm(items, ncols=80), start=1):
            file_name = item["image"]["file_name"]
            image_path = image_root / file_name
            min_area = int(item["min_area"])
            stem = safe_name(Path(file_name).with_suffix("").as_posix())
            vis_name = f"{idx:04d}_minarea{min_area}_{stem}.png"
            save_path = save_subdir / vis_name
            title = (
                f"{ws}/{split}/{file_name}  "
                f"small_gt={len(item['small_anns'])}  min_area={min_area}"
            )
            save_visual(image_path, item, title, save_path, args.dpi)
    print("\nSmall GT counts by WS:")
    for ws in sorted(count_by_ws, key=lambda name: int(name[2:])):
        print(f"{ws}: {count_by_ws[ws]} GTs in {image_count_by_ws[ws]} images")


if __name__ == "__main__":
    main()
