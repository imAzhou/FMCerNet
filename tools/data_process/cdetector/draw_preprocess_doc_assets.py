import argparse
import json
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy import sparse
from scipy.ndimage import find_objects


DATASET_ROOT = "/shared_storage/xzly/datasets/CervicalDatasets/ComparisonDetectorDataset"
OUTPUT_ROOT = "data_resource/CDetector_WS400"
PATCH_ROOT = "data_resource/CDetector_WS400/WS_400"
WORK_DIR = "work_dir/tmp/preprocess_cdetector"
WINDOW_SIZE = 400
STRIDE = WINDOW_SIZE - 50
MINLEN = 100
SAMPLE_MODE = "train"
SAMPLE_FILE = "00404.bmp"
SAMPLE_COORDS = [0, 0, 400, 400]
POSITIVE_CLASSES = ["AGC", "ASC-US", "LSIL", "ASC-H", "HSIL"]
CLASS_COLORS = {
    "AGC": "#9ECAE1",
    "ASC-US": "#FCAE91",
    "LSIL": "#FB6A4A",
    "ASC-H": "#DE2D26",
    "HSIL": "#A50F15",
}
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
EXPECTED_SAMPLE_GT = ["ASC-US", "LSIL", "ASC-H", "HSIL"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Draw CDetector preprocessing LaTeX table and workflow figure."
    )
    parser.add_argument("--dataset-root", default=DATASET_ROOT)
    parser.add_argument("--output-root", default=OUTPUT_ROOT)
    parser.add_argument("--patch-root", default=PATCH_ROOT)
    parser.add_argument("--work-dir", default=WORK_DIR)
    parser.add_argument("--window-size", type=int, default=WINDOW_SIZE)
    parser.add_argument("--stride", type=int, default=STRIDE)
    parser.add_argument("--minlen", type=int, default=MINLEN)
    return parser.parse_args()


def ensure_file(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(path)


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


def load_sparse_mask(npz_path):
    loader = np.load(npz_path)
    sparse_mask = sparse.coo_matrix(
        (loader["data"], (loader["row"], loader["col"])),
        shape=loader["shape"],
    )
    return sparse_mask.toarray().astype(np.int16)


def count_valid_instances(dataset_root):
    counts = {mode: {name: 0 for name in CLSNAME_MAP} for mode in ["train", "test"]}
    for mode in ["train", "test"]:
        json_path = os.path.join(dataset_root, f"{mode}.json")
        ensure_file(json_path)
        with open(json_path, "r", encoding="utf-8") as f:
            json_data = json.load(f)

        cat_id_to_name = {item["id"]: item["name"] for item in json_data["categories"]}
        for ann in json_data["annotations"]:
            raw_name = cat_id_to_name[ann["category_id"]]
            if raw_name not in CLSNAME_MAP:
                raise KeyError(f"Unexpected category name: {raw_name}")
            _, _, w, h = ann["bbox"]
            if w > 5 and h > 5:
                counts[mode][raw_name] += 1
    return counts


def write_latex_table(counts, save_path):
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{CDetector category mapping and valid annotation counts. Valid annotations follow the preprocessing filter $w>5$ and $h>5$; SAM-based mask generation and patch labels keep only AGC, ASC-US, LSIL, ASC-H, and HSIL.}",
        r"\label{tab:cdetector_clsname_map}",
        r"\begin{tabular}{llrr}",
        r"\hline",
        r"Original class & Mapped class & Train instances & Test instances \\",
        r"\hline",
    ]
    for raw_name, mapped_name in CLSNAME_MAP.items():
        lines.append(
            f"{raw_name} & {mapped_name} & "
            f"{counts['train'][raw_name]} & {counts['test'][raw_name]} \\\\"
        )
    train_total = sum(counts["train"].values())
    test_total = sum(counts["test"].values())
    lines.extend([
        r"\hline",
        f"Total & -- & {train_total} & {test_total} \\\\",
    ])
    lines.extend([
        r"\hline",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ])
    with open(save_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def load_mode_items(dataset_root, mode):
    json_path = os.path.join(dataset_root, f"{mode}.json")
    ensure_file(json_path)
    with open(json_path, "r", encoding="utf-8") as f:
        json_data = json.load(f)

    cat_id_to_name = {
        item["id"]: CLSNAME_MAP[item["name"]]
        for item in json_data["categories"]
    }
    image_id_to_anns = defaultdict(list)
    for ann in json_data["annotations"]:
        mapped_name = cat_id_to_name[ann["category_id"]]
        _, _, w, h = ann["bbox"]
        if w <= 5 or h <= 5 or mapped_name not in POSITIVE_CLASSES:
            continue
        new_ann = dict(ann)
        new_ann["mapped_name"] = mapped_name
        image_id_to_anns[ann["image_id"]].append(new_ann)

    items = []
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


def draw_labeled_boxes(ax, boxes, labels, colors, linewidth=2.0, fontsize=10):
    for box, label, color in zip(boxes, labels, colors):
        x1, y1, x2, y2 = box
        ax.add_patch(
            patches.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                linewidth=linewidth,
                edgecolor=color,
                facecolor="none",
            )
        )
        ax.text(
            x1,
            max(0, y1 - 4),
            label,
            color="white",
            fontsize=fontsize,
            va="bottom",
            bbox={"facecolor": color, "edgecolor": "none", "alpha": 0.90, "pad": 1.5},
        )


def draw_mask_overlay(ax, masks, colors, alpha=0.35):
    for mask, color in zip(masks, colors):
        if not mask.any():
            continue
        rgb = matplotlib.colors.to_rgb(color)
        rgba = np.zeros((*mask.shape, 4), dtype=np.float32)
        rgba[..., :3] = rgb
        rgba[..., 3] = mask.astype(np.float32) * alpha
        ax.imshow(rgba)
        ax.contour(mask, levels=[0.5], colors=[color], linewidths=1.2)


def save_panel_input(image, item, colors, save_path):
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.imshow(image)
    boxes = []
    labels = []
    box_colors = []
    for inst_id, ann in enumerate(item["annos"], start=1):
        x, y, w, h = ann["bbox"]
        boxes.append([x, y, x + w, y + h])
        labels.append(ann["mapped_name"])
        box_colors.append(colors[ann["mapped_name"]])
    draw_labeled_boxes(ax, boxes, labels, box_colors, linewidth=1.8, fontsize=8)
    ax.set_title("Input ROI with bbox prompts and mapped classes", fontsize=12)
    ax.axis("off")
    fig.tight_layout(pad=0.3)
    fig.savefig(save_path, dpi=220, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def save_panel_sam(image, item, roi_mask, colors, save_path):
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.imshow(image)
    masks = []
    boxes = []
    labels = []
    box_colors = []
    for inst_id, ann in enumerate(item["annos"], start=1):
        x, y, w, h = ann["bbox"]
        masks.append(roi_mask == inst_id)
        boxes.append([x, y, x + w, y + h])
        labels.append(ann["mapped_name"])
        box_colors.append(colors[ann["mapped_name"]])
    draw_mask_overlay(ax, masks, box_colors, alpha=0.32)
    draw_labeled_boxes(ax, boxes, labels, box_colors, linewidth=1.4, fontsize=8)
    ax.set_title("SAM segmentation masks from bbox prompts", fontsize=12)
    ax.axis("off")
    fig.tight_layout(pad=0.3)
    fig.savefig(save_path, dpi=220, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def save_panel_windows(image, item, roi_mask, cut_points, selected_coords, colors, save_path):
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.imshow(image)
    positive_coords = []
    for patch_coords in cut_points:
        _, labels, _ = calc_patch_anns(patch_coords, item["annos"], roi_mask)
        if labels:
            positive_coords.append(patch_coords)

    for patch_coords in cut_points:
        x1, y1, x2, y2 = patch_coords
        ax.add_patch(
            patches.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                linewidth=0.8,
                edgecolor=(1.0, 1.0, 1.0, 0.55),
                facecolor="none",
            )
        )
    for patch_coords in positive_coords:
        x1, y1, x2, y2 = patch_coords
        ax.add_patch(
            patches.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                linewidth=1.8,
                edgecolor="#2ca02c",
                facecolor=(0.17, 0.63, 0.17, 0.13),
            )
        )
    x1, y1, x2, y2 = selected_coords
    ax.add_patch(
        patches.Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            linewidth=3.0,
            edgecolor="#d62728",
            facecolor=(0.84, 0.15, 0.16, 0.12),
        )
    )
    ax.text(
        x1 + 8,
        y1 + 22,
        "selected positive window",
        color="white",
        fontsize=9,
        bbox={"facecolor": "#d62728", "edgecolor": "none", "alpha": 0.90, "pad": 2},
    )
    ax.set_title("Fixed 400 x 400 window cutting; retained-mask windows are positive", fontsize=12)
    ax.axis("off")
    fig.tight_layout(pad=0.3)
    fig.savefig(save_path, dpi=220, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def save_panel_patch(image, item, roi_mask, selected_coords, window_size, colors, save_path):
    patch_img = cut_img(image, selected_coords, window_size)
    bboxes, labels, masks = calc_patch_anns(selected_coords, item["annos"], roi_mask)
    gt_names = sorted(set(labels), key=POSITIVE_CLASSES.index)
    if gt_names != EXPECTED_SAMPLE_GT:
        raise RuntimeError(f"Unexpected sample GT: {gt_names}")

    fig, ax = plt.subplots(figsize=(5.4, 5.4))
    ax.imshow(patch_img)
    box_colors = [colors[label] for label in labels]
    draw_mask_overlay(ax, masks, box_colors, alpha=0.34)
    draw_labeled_boxes(ax, bboxes, labels, box_colors, linewidth=1.7, fontsize=8)
    ax.set_title("Patch GT = {" + ", ".join(gt_names) + "}", fontsize=12)
    ax.axis("off")
    fig.tight_layout(pad=0.3)
    fig.savefig(save_path, dpi=220, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    return gt_names


def save_flow_figure(
    image,
    item,
    roi_mask,
    cut_points,
    selected_coords,
    window_size,
    colors,
    save_path_png,
):
    fig, axes = plt.subplots(
        1,
        4,
        figsize=(20, 5.2),
        gridspec_kw={"width_ratios": [1.25, 1.25, 1.25, 0.95]},
    )

    axes[0].imshow(image)
    boxes = []
    labels = []
    box_colors = []
    for inst_id, ann in enumerate(item["annos"], start=1):
        x, y, w, h = ann["bbox"]
        boxes.append([x, y, x + w, y + h])
        labels.append(ann["mapped_name"])
        box_colors.append(colors[ann["mapped_name"]])
    draw_labeled_boxes(axes[0], boxes, labels, box_colors, linewidth=1.5, fontsize=7)
    axes[0].set_title("Original + bbox + class", fontsize=12)

    axes[1].imshow(image)
    masks = [roi_mask == inst_id for inst_id in range(1, len(item["annos"]) + 1)]
    draw_mask_overlay(axes[1], masks, box_colors, alpha=0.32)
    draw_labeled_boxes(axes[1], boxes, labels, box_colors, linewidth=1.1, fontsize=7)
    axes[1].set_title("SAM masks", fontsize=12)

    axes[2].imshow(image)
    positive_coords = []
    for patch_coords in cut_points:
        _, patch_labels, _ = calc_patch_anns(patch_coords, item["annos"], roi_mask)
        if patch_labels:
            positive_coords.append(patch_coords)
    for patch_coords in cut_points:
        x1, y1, x2, y2 = patch_coords
        axes[2].add_patch(
            patches.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                linewidth=0.65,
                edgecolor=(1.0, 1.0, 1.0, 0.6),
                facecolor="none",
            )
        )
    for patch_coords in positive_coords:
        x1, y1, x2, y2 = patch_coords
        axes[2].add_patch(
            patches.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                linewidth=1.4,
                edgecolor="#2ca02c",
                facecolor=(0.17, 0.63, 0.17, 0.13),
            )
        )
    x1, y1, x2, y2 = selected_coords
    axes[2].add_patch(
        patches.Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            linewidth=2.5,
            edgecolor="#d62728",
            facecolor=(0.84, 0.15, 0.16, 0.12),
        )
    )
    axes[2].set_title("Fixed-window cutting", fontsize=12)

    patch_img = cut_img(image, selected_coords, window_size)
    bboxes, patch_labels, patch_masks = calc_patch_anns(selected_coords, item["annos"], roi_mask)
    gt_names = sorted(set(patch_labels), key=POSITIVE_CLASSES.index)
    if gt_names != EXPECTED_SAMPLE_GT:
        raise RuntimeError(f"Unexpected sample GT: {gt_names}")
    axes[3].imshow(patch_img)
    patch_colors = [colors[label] for label in patch_labels]
    draw_mask_overlay(axes[3], patch_masks, patch_colors, alpha=0.34)
    draw_labeled_boxes(axes[3], bboxes, patch_labels, patch_colors, linewidth=1.3, fontsize=7)
    axes[3].set_title("Multi-label GT", fontsize=12)
    axes[3].text(
        0.5,
        -0.06,
        "{" + ", ".join(gt_names) + "}",
        transform=axes[3].transAxes,
        ha="center",
        va="top",
        fontsize=11,
        color="#111111",
    )

    for ax in axes:
        ax.axis("off")

    fig.suptitle("CDetector preprocessing workflow", fontsize=16, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.94], w_pad=1.3)
    fig.savefig(save_path_png, dpi=240, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def load_sample_patch_record(patch_root):
    patch_json_path = os.path.join(patch_root, "annofiles", "patches_train.json")
    ensure_file(patch_json_path)
    with open(patch_json_path, "r", encoding="utf-8") as f:
        patch_items = json.load(f)
    for item in patch_items:
        extra_info = item["extra_info"]
        if (
            extra_info["mode"] == SAMPLE_MODE
            and extra_info["source_file"] == SAMPLE_FILE
            and extra_info["square_coords"] == SAMPLE_COORDS
        ):
            return item
    raise RuntimeError(f"Missing sample patch record for {SAMPLE_MODE}/{SAMPLE_FILE} {SAMPLE_COORDS}")


def main():
    args = parse_args()
    if os.path.isabs(args.work_dir) and not args.work_dir.startswith(
        os.path.abspath("work_dir/tmp")
    ):
        raise ValueError(f"work_dir must stay under work_dir/tmp: {args.work_dir}")

    os.makedirs(args.work_dir, exist_ok=True, mode=0o777)
    component_dir = os.path.join(args.work_dir, "components")
    os.makedirs(component_dir, exist_ok=True, mode=0o777)

    counts = count_valid_instances(args.dataset_root)
    table_path = os.path.join(args.work_dir, "clsname_map_table.tex")
    write_latex_table(counts, table_path)

    sample_patch = load_sample_patch_record(args.patch_root)
    sample_gt = [POSITIVE_CLASSES[idx] for idx in sample_patch["gt_label"]]
    if sample_gt != EXPECTED_SAMPLE_GT:
        raise RuntimeError(f"Unexpected patch-json GT: {sample_gt}")

    sample_items = load_mode_items(args.dataset_root, SAMPLE_MODE)
    sample_item = next(item for item in sample_items if item["file_name"] == SAMPLE_FILE)
    image_path = os.path.join(args.dataset_root, SAMPLE_MODE, SAMPLE_FILE)
    mask_path = os.path.join(
        args.output_root,
        "roi_inst_mask",
        SAMPLE_MODE,
        os.path.splitext(SAMPLE_FILE)[0] + ".npz",
    )
    ensure_file(image_path)
    ensure_file(mask_path)

    image = Image.open(image_path).convert("RGB")
    roi_mask = load_sparse_mask(mask_path)
    cut_points = generate_cut_regions(
        (0, 0),
        image.size[0],
        image.size[1],
        args.window_size,
        args.stride,
        minlen=args.minlen,
    )
    if SAMPLE_COORDS not in cut_points:
        raise RuntimeError(f"Sample coords are not generated by current window settings: {SAMPLE_COORDS}")

    colors = CLASS_COLORS
    save_panel_input(
        image,
        sample_item,
        colors,
        os.path.join(component_dir, "01_input_bbox_label.png"),
    )
    save_panel_sam(
        image,
        sample_item,
        roi_mask,
        colors,
        os.path.join(component_dir, "02_sam_mask.png"),
    )
    save_panel_windows(
        image,
        sample_item,
        roi_mask,
        cut_points,
        SAMPLE_COORDS,
        colors,
        os.path.join(component_dir, "03_fixed_window_cutting.png"),
    )
    save_panel_patch(
        image,
        sample_item,
        roi_mask,
        SAMPLE_COORDS,
        args.window_size,
        colors,
        os.path.join(component_dir, "04_positive_patch_gt.png"),
    )
    save_flow_figure(
        image,
        sample_item,
        roi_mask,
        cut_points,
        SAMPLE_COORDS,
        args.window_size,
        colors,
        os.path.join(args.work_dir, "preprocess_flow.png"),
    )

    print(f"Saved LaTeX table: {table_path}")
    print(f"Saved workflow figure: {os.path.join(args.work_dir, 'preprocess_flow.png')}")
    print(f"Saved component panels: {component_dir}")


if __name__ == "__main__":
    main()
