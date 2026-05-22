import argparse
import json
import os
import re
import shutil
from collections import defaultdict

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy import sparse
from scipy.ndimage import find_objects


DATASET_ROOT = "/shared_storage/xzly/datasets/CervicalDatasets/ComparisonDetectorDataset"
OUTPUT_ROOT = "work_dir/data_process/cdetector/patch_visual"
WINDOW_SIZE = 400
STRIDE = WINDOW_SIZE - 50
POSITIVE_CLASSES = ["AGC", "ASC-US", "LSIL", "ASC-H", "HSIL"]
CLSNAME_MAP = {
    "ascus": "ASC-US",
    "lsil": "LSIL",
    "asch": "ASC-H",
    "hsil": "HSIL",
    "scc": "HSIL",
    "agc": "AGC",
    "trichomonas": "NILM",
    "candida": "NILM",
    "flora": "NILM",
    "herps": "NILM",
    "actinomyces": "NILM",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize sampled CDetector WS400 patches.")
    parser.add_argument("--dataset-root", default=DATASET_ROOT)
    parser.add_argument("--output-root", default=OUTPUT_ROOT)
    parser.add_argument("--modes", nargs="+", default=["train", "test"], choices=["train", "test"])
    parser.add_argument("--mask-root", default=None)
    parser.add_argument("--sample-size", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--positive-only", action="store_true")
    parser.add_argument("--clean-output", action="store_true", help="Remove the visualization directory before writing PNGs.")
    parser.add_argument("--window-size", type=int, default=WINDOW_SIZE)
    parser.add_argument("--stride", type=int, default=STRIDE)
    parser.add_argument("--minlen", type=int, default=100)
    return parser.parse_args()


def safe_name(name):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def generate_cut_regions(region_start, region_width, region_height, k, stride=400, minlen=0):
    x_start, y_start = region_start
    overlap = k - stride
    cut_regions = []
    exact_w = (region_width // stride) * stride + overlap
    exact_h = (region_height // stride) * stride + overlap
    w_rem = region_width - exact_w
    h_rem = region_height - exact_h
    end_x = exact_w if w_rem > minlen else exact_w - stride
    end_y = exact_h if h_rem > minlen else exact_h - stride

    for x in range(0, end_x, stride):
        for y in range(0, end_y, stride):
            x1, y1 = x, y
            x2, y2 = x1 + k, y1 + k
            if x2 > region_width:
                x2 = region_width
                x1 = x2 - k
            if y2 > region_height:
                y2 = region_height
                y1 = y2 - k
            cut_regions.append([x1 + x_start, y1 + y_start, x2 + x_start, y2 + y_start])
    return cut_regions


def load_items(dataset_root, modes):
    items = []
    for mode in modes:
        with open(os.path.join(dataset_root, f"{mode}.json"), "r", encoding="utf-8") as f:
            json_data = json.load(f)

        cat_id_to_name = {
            item["id"]: CLSNAME_MAP[item["name"]]
            for item in json_data["categories"]
        }
        image_id_to_anns = defaultdict(list)
        for ann in json_data["annotations"]:
            mapped_name = cat_id_to_name[ann["category_id"]]
            x, y, w, h = ann["bbox"]
            if w <= 5 or h <= 5 or mapped_name not in POSITIVE_CLASSES:
                continue
            new_ann = dict(ann)
            new_ann["mapped_name"] = mapped_name
            image_id_to_anns[ann["image_id"]].append(new_ann)

        for img in json_data["images"]:
            annos = sorted(
                image_id_to_anns.get(img["id"], []),
                key=lambda ann: ann["bbox"][2] * ann["bbox"][3],
                reverse=True,
            )
            items.append({
                "mode": mode,
                "file_name": img["file_name"],
                "width": img["width"],
                "height": img["height"],
                "annos": annos,
            })
    return items


def sample_items(items, sample_size, seed, positive_only):
    if positive_only:
        items = [item for item in items if item["annos"]]
    rng = np.random.default_rng(seed)
    sample_size = min(sample_size, len(items))
    indices = rng.choice(len(items), size=sample_size, replace=False)
    return [items[i] for i in indices]


def load_sparse_mask(npz_path):
    loader = np.load(npz_path)
    sparse_mask = sparse.coo_matrix(
        (loader["data"], (loader["row"], loader["col"])),
        shape=loader["shape"],
    )
    return sparse_mask.toarray().astype(np.int16)


def cut_img(roi_img, patch_coords, window_size):
    patch = Image.new("RGB", (window_size, window_size), color=(255, 255, 255))
    rw, rh = roi_img.size
    x1, y1, x2, y2 = patch_coords
    int_x1 = max(0, x1)
    int_y1 = max(0, y1)
    int_x2 = min(x2, rw)
    int_y2 = min(y2, rh)
    cropped = roi_img.crop((int_x1, int_y1, int_x2, int_y2))
    patch.paste(cropped, (int_x1 - x1, int_y1 - y1))
    return patch


def calc_patch_anns(patch_coords, annlist, roi_mask):
    rpx1, rpy1, rpx2, rpy2 = patch_coords
    patch_mask = roi_mask[rpy1:rpy2, rpx1:rpx2]

    ann_bboxes = []
    ann_names = []
    ann_masks = []
    annidx = np.unique(patch_mask)
    if len(annidx) <= 1:
        return ann_bboxes, ann_names, ann_masks

    objects = find_objects(patch_mask)
    for aidx in annidx[1:]:
        obj_slice = objects[aidx - 1]
        if obj_slice is None:
            continue
        yslice, xslice = obj_slice
        by1, by2 = yslice.start, yslice.stop
        bx1, bx2 = xslice.start, xslice.stop
        bwidth, bheight = bx2 - bx1, by2 - by1
        is_small = min(bwidth, bheight) < 50
        is_near_edge = (
            bx1 <= 1
            or by1 <= 1
            or bx2 >= patch_mask.shape[1] - 1
            or by2 >= patch_mask.shape[0] - 1
        )
        if is_small and is_near_edge:
            continue

        ann_bboxes.append([bx1, by1, bx2, by2])
        ann_names.append(annlist[aidx - 1]["mapped_name"])
        ann_masks.append(patch_mask == aidx)

    return ann_bboxes, ann_names, ann_masks


def draw_instances(ax, masks, boxes, labels, alpha=0.35):
    cmap = plt.get_cmap("tab20")
    for idx, (mask, box, label) in enumerate(zip(masks, boxes, labels), start=1):
        color = cmap((idx - 1) % 20)
        if mask.any():
            overlay = np.zeros((*mask.shape, 4), dtype=np.float32)
            overlay[..., :3] = color[:3]
            overlay[..., 3] = mask.astype(np.float32) * alpha
            ax.imshow(overlay)
            ax.contour(mask, levels=[0.5], colors=[color], linewidths=1)

        x1, y1, x2, y2 = box
        ax.add_patch(
            patches.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                linewidth=1.5,
                edgecolor=color,
                facecolor="none",
            )
        )
        ax.text(
            x1,
            max(0, y1 - 3),
            label,
            color="white",
            fontsize=8,
            va="bottom",
            bbox={"facecolor": color, "edgecolor": "none", "alpha": 0.85, "pad": 1},
        )


def save_original_visual(image, item, roi_mask, save_path):
    boxes = []
    labels = []
    masks = []
    for inst_id, ann in enumerate(item["annos"], start=1):
        x, y, w, h = ann["bbox"]
        boxes.append([x, y, x + w, y + h])
        labels.append(ann["mapped_name"])
        masks.append(roi_mask == inst_id)

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(image)
    draw_instances(ax, masks, boxes, labels)
    ax.axis("off")
    fig.tight_layout(pad=0)
    fig.savefig(save_path, dpi=180, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def save_patch_visual(patch_img, bboxes, labels, masks, save_path):
    multilabel_names = sorted(set(labels))
    title = "multilabel: " + (", ".join(multilabel_names) if multilabel_names else "None")
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(patch_img)
    draw_instances(ax, masks, bboxes, labels)
    ax.set_title(title, fontsize=11)
    ax.axis("off")
    fig.tight_layout(pad=0.4)
    fig.savefig(save_path, dpi=180, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def visualize_item(args, item):
    purename = os.path.splitext(item["file_name"])[0]
    folder = os.path.join(args.output_root, f"{item['mode']}_{safe_name(purename)}")
    os.makedirs(folder, exist_ok=True, mode=0o777)

    image_path = os.path.join(args.dataset_root, item["mode"], item["file_name"])
    image = Image.open(image_path).convert("RGB")
    mask_root = args.mask_root or args.output_root
    mask_path = os.path.join(mask_root, "roi_inst_mask", item["mode"], f"{purename}.npz")
    roi_mask = load_sparse_mask(mask_path)

    save_original_visual(image, item, roi_mask, os.path.join(folder, "origin_bbox_mask_label.png"))

    cut_points = generate_cut_regions(
        (0, 0),
        image.size[0],
        image.size[1],
        args.window_size,
        args.stride,
        minlen=args.minlen,
    )
    for patch_idx, patch_coords in enumerate(cut_points):
        patch_img = cut_img(image, patch_coords, args.window_size)
        bboxes, labels, masks = calc_patch_anns(patch_coords, item["annos"], roi_mask)
        multilabel = "_".join(sorted(set(labels))) if labels else "None"
        filename = f"patch_{patch_idx:03d}_{safe_name(multilabel)}.png"
        save_patch_visual(patch_img, bboxes, labels, masks, os.path.join(folder, filename))


def main():
    args = parse_args()
    if args.clean_output and os.path.exists(args.output_root):
        shutil.rmtree(args.output_root)
    items = load_items(args.dataset_root, args.modes)
    sampled_items = sample_items(items, args.sample_size, args.seed, args.positive_only)
    os.makedirs(args.output_root, exist_ok=True, mode=0o777)
    for item in sampled_items:
        visualize_item(args, item)
    print(f"Saved visualizations for {len(sampled_items)} images to {args.output_root}")


if __name__ == "__main__":
    main()
