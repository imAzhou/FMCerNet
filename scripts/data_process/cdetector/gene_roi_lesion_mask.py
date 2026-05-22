import argparse
import json
import os
from collections import defaultdict

import numpy as np
import torch
import torch.distributed as dist
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from scipy import sparse
from tqdm import tqdm

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


DATASET_ROOT = "/shared_storage/xzly/datasets/CervicalDatasets/ComparisonDetectorDataset"
OUTPUT_ROOT = "data_resource/CDetector_WS400"
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
    parser = argparse.ArgumentParser(description="Generate CDetector ROI lesion masks with SAM2.")
    parser.add_argument("--dataset-root", default=DATASET_ROOT)
    parser.add_argument("--output-root", default=OUTPUT_ROOT)
    parser.add_argument("--modes", nargs="+", default=["train", "test"], choices=["train", "test"])
    parser.add_argument("--sam2-checkpoint", default="checkpoints/sam2.1_hiera_large.pt")
    parser.add_argument("--model-cfg", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--limit", type=int, default=None, help="Optional smoke-test limit per mode.")
    parser.add_argument("--positive-only", action="store_true", help="Only keep images with positive annotations.")
    parser.add_argument("--sample-size", type=int, default=None, help="Randomly sample N images after loading modes.")
    parser.add_argument("--vis-dir", default=None, help="Optional directory to save mask visualizations.")
    return parser.parse_args()


def init_distributed():
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        return 0, 1, 0, False

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=world_size,
        rank=rank,
    )
    return rank, world_size, local_rank, True


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_mode_items(dataset_root, mode, limit=None):
    json_path = os.path.join(dataset_root, f"{mode}.json")
    with open(json_path, "r", encoding="utf-8") as f:
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

    if limit is not None:
        items = items[:limit]
    return items


def save_sparse_mask(mask, save_path):
    sparse_mask = sparse.coo_matrix(mask)
    np.savez_compressed(
        save_path,
        data=sparse_mask.data,
        row=sparse_mask.row,
        col=sparse_mask.col,
        shape=sparse_mask.shape,
    )


def save_visualization(image, annos, roi_mask, save_path):
    image_array = np.array(image.convert("RGB"))
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(image_array)

    cmap = plt.get_cmap("tab20")
    for inst_id, ann in enumerate(annos, start=1):
        inst_mask = roi_mask == inst_id
        color = cmap((inst_id - 1) % 20)
        if inst_mask.any():
            overlay = np.zeros((*inst_mask.shape, 4), dtype=np.float32)
            overlay[..., :3] = color[:3]
            overlay[..., 3] = inst_mask.astype(np.float32) * 0.35
            ax.imshow(overlay)
            ax.contour(inst_mask, levels=[0.5], colors=[color], linewidths=1)

        x, y, w, h = ann["bbox"]
        ax.add_patch(
            patches.Rectangle(
                (x, y),
                w,
                h,
                linewidth=1.5,
                edgecolor=color,
                facecolor="none",
            )
        )
        ax.text(
            x,
            max(0, y - 3),
            ann["mapped_name"],
            color="white",
            fontsize=8,
            va="bottom",
            bbox={"facecolor": color, "edgecolor": "none", "alpha": 0.85, "pad": 1},
        )

    ax.axis("off")
    fig.tight_layout(pad=0)
    os.makedirs(os.path.dirname(save_path), exist_ok=True, mode=0o777)
    fig.savefig(save_path, dpi=180, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def build_predictor(args, device):
    if torch.cuda.get_device_properties(device).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    sam2_model = build_sam2(args.model_cfg, args.sam2_checkpoint, device=device)
    return SAM2ImagePredictor(sam2_model)


def infer_rank_items(args, rank, items, device):
    predictor = build_predictor(args, device)
    processed = 0
    skipped = 0
    empty_mask = 0

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        pbar = tqdm(items, ncols=80, disable=rank != 0)
        for item in pbar:
            purename = os.path.splitext(item["file_name"])[0]
            mask_save_dir = os.path.join(args.output_root, "roi_inst_mask", item["mode"])
            os.makedirs(mask_save_dir, exist_ok=True, mode=0o777)
            mask_save_path = os.path.join(mask_save_dir, f"{purename}.npz")
            if os.path.exists(mask_save_path):
                skipped += 1
                continue

            img_path = os.path.join(args.dataset_root, item["mode"], item["file_name"])
            image = Image.open(img_path).convert("RGB")
            width, height = image.size
            roi_mask = np.zeros((height, width), dtype=np.int16)

            input_boxes = []
            for ann in item["annos"]:
                x, y, w, h = ann["bbox"]
                input_boxes.append([x, y, x + w, y + h])

            if input_boxes:
                image_array = np.array(image)
                predictor.set_image(image_array)
                masks, _, _ = predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=np.asarray(input_boxes),
                    multimask_output=False,
                )
                if len(masks.shape) == 3:
                    masks = masks[None, :]
                masks = masks.squeeze(1)

                for inst_id, mask in enumerate(masks, start=1):
                    if np.sum(mask) == 0:
                        empty_mask += 1
                    ys, xs = np.where(mask)
                    roi_mask[ys, xs] = inst_id

            save_sparse_mask(roi_mask, mask_save_path)
            if args.vis_dir:
                vis_name = f"{item['mode']}_{purename}.png"
                save_visualization(image, item["annos"], roi_mask, os.path.join(args.vis_dir, vis_name))
            processed += 1

    print(
        f"[rank {rank}] processed={processed}, skipped={skipped}, "
        f"empty_mask={empty_mask}",
        flush=True,
    )


def main():
    args = parse_args()
    rank, world_size, local_rank, distributed = init_distributed()
    set_seed(args.seed + rank)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for SAM2 mask generation.")
    device = torch.device(f"cuda:{local_rank}")

    all_items = []
    for mode in args.modes:
        all_items.extend(load_mode_items(args.dataset_root, mode, args.limit))
    if args.positive_only:
        all_items = [item for item in all_items if item["annos"]]
    if args.sample_size is not None:
        rng = np.random.default_rng(args.seed)
        sample_size = min(args.sample_size, len(all_items))
        sample_indices = rng.choice(len(all_items), size=sample_size, replace=False)
        all_items = [all_items[i] for i in sample_indices]

    rank_items = all_items[rank::world_size]
    print(
        f"[rank {rank}] total_items={len(all_items)}, rank_items={len(rank_items)}",
        flush=True,
    )
    infer_rank_items(args, rank, rank_items, device)

    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
