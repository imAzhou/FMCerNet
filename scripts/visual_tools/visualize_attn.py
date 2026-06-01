import argparse
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from mmcv.transforms import Compose
from mmengine.config import Config
from mmengine.registry import init_default_scope
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fmcernet.nets import PatchNet

CLASS_NAMES = ["AGC", "ASC-US", "LSIL", "ASC-H", "HSIL"]
PRED_INFO_KEYS = ["case_type", "img_prob", "pos_prob", "img_pred_pn", "img_pred_mcls"]
FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
CASE_TYPES = {"TP", "FP", "TN", "FN"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize WSCerMLC taskhead attn from a COCO-format json."
    )
    parser.add_argument("json_file", type=Path)
    parser.add_argument("config_file", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("work_dir/tmp/taskhead_attn_vis"),
    )
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-items", type=int)
    parser.add_argument(
        "--image-ids",
        type=str,
        help="Comma-separated COCO image ids to visualize.",
    )
    return parser.parse_args()


def load_coco(json_file):
    with json_file.open("r") as f:
        coco = json.load(f)

    for key in ["categories", "images", "annotations"]:
        if key not in coco:
            raise KeyError(f"COCO json must contain key: {key}")

    categories = {}
    for category in coco["categories"]:
        for key in ["id", "name", "color"]:
            if key not in category:
                raise KeyError(f"Category must contain key: {key}")
        categories[category["id"]] = {
            "name": category["name"],
            "color": tuple(int(v) for v in category["color"]),
        }

    anns_by_image = defaultdict(list)
    for ann in coco["annotations"]:
        anns_by_image[ann["image_id"]].append(ann)

    for image_info in coco["images"]:
        for key in PRED_INFO_KEYS:
            if key not in image_info:
                raise KeyError(f"Image item must contain prediction key: {key}")

    return coco["images"], anns_by_image, categories


def select_images(images, image_ids, max_items):
    selected = images
    if image_ids:
        wanted = {int(item) for item in image_ids.split(",") if item}
        selected = [item for item in selected if int(item["id"]) in wanted]
    if max_items is not None:
        selected = selected[:max_items]
    return selected


def resolve_image_root(cfg, image_root):
    if image_root is not None:
        return image_root
    data_root = Path(cfg.data_root)
    data_prefix = Path(cfg.val_datasets["data_prefix"])
    return data_root / data_prefix


def build_model(cfg, checkpoint, device):
    if cfg.net_type != "patch":
        raise ValueError(f"Only patch net is supported, got cfg.net_type={cfg.net_type}")
    if cfg.taskhead_model != "wscer_mlc":
        raise ValueError(
            f"Only wscer_mlc taskhead is supported, got {cfg.taskhead_model}"
        )
    cfg.save_result_dir = None
    cfg.backbone_cfg["backbone_ckpt"] = None

    model = PatchNet(cfg).to(device)
    model.load_ckpt(str(checkpoint))
    model.eval()
    return model


def build_pipeline(cfg):
    init_default_scope("mmpretrain")
    return Compose(cfg.val_datasets["pipeline"])


def build_data_item(image_info, image_root, pipeline):
    data_info = dict(image_info)
    data_info["img_path"] = str(image_root / image_info["file_name"])
    data_info["gt_label"] = []
    return pipeline(data_info)


def collate_batch(items):
    return {
        "inputs": torch.stack([item["inputs"] for item in items], dim=0),
        "data_samples": [item["data_samples"] for item in items],
    }


def infer_attn(model, batch, device):
    batch["inputs"] = batch["inputs"].to(device)
    with torch.no_grad():
        outputs = model(batch, "predict")
    return [item.attn.detach().cpu() for item in outputs]


def normalize_attn(attn):
    attn = attn.float().numpy()
    side = int(math.sqrt(attn.shape[0]))
    if side * side != attn.shape[0]:
        raise ValueError(f"Attn token count must be square, got {attn.shape[0]}")
    attn = attn.reshape(side, side)
    attn_min = float(attn.min())
    attn_max = float(attn.max())
    if attn_max == attn_min:
        return np.zeros_like(attn, dtype=np.float32)
    return ((attn - attn_min) / (attn_max - attn_min)).astype(np.float32)


def colorize_attn(attn, size):
    heat = Image.fromarray((attn * 255).astype(np.uint8)).resize(size, Image.BILINEAR)
    arr = np.asarray(heat, dtype=np.float32) / 255.0
    red = np.clip(1.5 * arr, 0.0, 1.0)
    green = np.clip(1.5 - np.abs(arr - 0.55) * 2.0, 0.0, 1.0)
    blue = np.clip(1.0 - 1.8 * arr, 0.0, 1.0)
    alpha = np.clip(0.15 + 0.55 * arr, 0.0, 0.65)
    rgba = np.stack([red, green, blue, alpha], axis=-1)
    return Image.fromarray((rgba * 255).astype(np.uint8), mode="RGBA")


def scaled_bbox(bbox, image_info, actual_size):
    x, y, w, h = [float(v) for v in bbox]
    scale_x = actual_size[0] / float(image_info["width"])
    scale_y = actual_size[1] / float(image_info["height"])
    return [
        x * scale_x,
        y * scale_y,
        (x + w) * scale_x,
        (y + h) * scale_y,
    ]


def draw_bboxes(image, image_info, annotations, categories):
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    for ann in annotations:
        category = categories[ann["category_id"]]
        color = category["color"]
        box = scaled_bbox(ann["bbox"], image_info, image.size)
        draw.rectangle(box, outline=color, width=3)

        label = category["name"]
        text_box = draw.textbbox((0, 0), label, font=font)
        text_w = text_box[2] - text_box[0]
        text_h = text_box[3] - text_box[1]
        x0 = max(0, int(box[0]))
        y0 = max(0, int(box[1]) - text_h - 4)
        draw.rectangle(
            [x0, y0, x0 + text_w + 6, y0 + text_h + 4],
            fill=color,
        )
        draw.text((x0 + 3, y0 + 2), label, fill=(255, 255, 255), font=font)


def load_text_font(size):
    return ImageFont.truetype(FONT_PATH, size)


def text_size(text, font):
    box = ImageDraw.Draw(Image.new("RGB", (1, 1))).textbbox((0, 0), text, font=font)
    return box[2] - box[0], box[3] - box[1]


def draw_header_text(draw, x, y, width, height, text, font):
    text_w, text_h = text_size(text, font)
    pad_x = 8
    if text_w + pad_x * 2 > width:
        raise ValueError(f"Header text is wider than panel: {text}")
    draw.text(
        (x + pad_x, y + (height - text_h) // 2),
        text,
        fill=(20, 20, 20),
        font=font,
    )


def format_gt_text(annotations, categories):
    cls_names = []
    for ann in annotations:
        name = categories[ann["category_id"]]["name"]
        if name not in cls_names:
            cls_names.append(name)
    return f"gt:{cls_names}"


def format_pred_text(image_info):
    pos_prob = image_info["pos_prob"]
    if len(pos_prob) != len(CLASS_NAMES):
        raise ValueError(f"pos_prob must contain {len(CLASS_NAMES)} values.")

    prob_text = ", ".join(
        f"{name}:{float(prob):.2f}" for name, prob in zip(CLASS_NAMES, pos_prob)
    )
    return f"img_prob:{float(image_info['img_prob']):.2f}, pos_prob:[{prob_text}]"


def make_visualization(image_path, image_info, annotations, categories, attn):
    original = Image.open(image_path).convert("RGB")
    left = original.copy()
    draw_bboxes(left, image_info, annotations, categories)

    attn_map = normalize_attn(attn)
    heatmap = colorize_attn(attn_map, original.size)
    right = Image.alpha_composite(original.convert("RGBA"), heatmap).convert("RGB")
    draw_bboxes(right, image_info, annotations, categories)

    font = load_text_font(14)
    header_h = 30
    canvas = Image.new(
        "RGB",
        (original.width * 2, original.height + header_h),
        (255, 255, 255),
    )
    draw = ImageDraw.Draw(canvas)
    draw.rectangle([0, 0, canvas.width, header_h - 1], fill=(245, 245, 245))
    draw.line([(0, header_h - 1), (canvas.width, header_h - 1)], fill=(210, 210, 210))
    draw.line([(original.width, 0), (original.width, header_h - 1)], fill=(210, 210, 210))
    draw_header_text(
        draw,
        0,
        0,
        original.width,
        header_h,
        format_gt_text(annotations, categories),
        font,
    )
    draw_header_text(
        draw,
        original.width,
        0,
        original.width,
        header_h,
        format_pred_text(image_info),
        font,
    )
    canvas.paste(left, (0, header_h))
    canvas.paste(right, (original.width, header_h))
    return canvas


def safe_output_name(image_info):
    stem = Path(image_info["file_name"]).with_suffix("").as_posix()
    safe_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("_")
    return f'{image_info["id"]}_{safe_stem}.png'


def output_path(out_dir, image_info):
    case_type = image_info["case_type"]
    if case_type not in CASE_TYPES:
        raise ValueError(f"case_type must be one of {sorted(CASE_TYPES)}, got {case_type}")
    case_dir = out_dir / case_type
    case_dir.mkdir(parents=True, exist_ok=True)
    return case_dir / safe_output_name(image_info)


def main():
    args = parse_args()
    if str(args.out_dir).startswith("/tmp"):
        raise ValueError("Temporary outputs must be saved under work_dir/tmp, not /tmp")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    cfg = Config.fromfile(str(args.config_file))
    image_root = resolve_image_root(cfg, args.image_root)
    images, anns_by_image, categories = load_coco(args.json_file)
    selected_images = select_images(images, args.image_ids, args.max_items)

    device = torch.device(args.device)
    pipeline = build_pipeline(cfg)
    model = build_model(cfg, args.checkpoint, device)

    for start in tqdm(range(0, len(selected_images), args.batch_size), ncols=80):
        batch_infos = selected_images[start:start + args.batch_size]
        data_items = [build_data_item(info, image_root, pipeline) for info in batch_infos]
        attns = infer_attn(model, collate_batch(data_items), device)

        for image_info, attn in zip(batch_infos, attns):
            image_path = image_root / image_info["file_name"]
            canvas = make_visualization(
                image_path,
                image_info,
                anns_by_image[image_info["id"]],
                categories,
                attn,
            )
            canvas.save(output_path(args.out_dir, image_info))


if __name__ == "__main__":
    main()


'''
python scripts/visual_tools/visualize_attn.py \
  work_dir/tmp/pred_cases_400_coco.json \
  work_dir/mlc/ours/ws800/config.py \
  work_dir/mlc/ours/ws800/checkpoints/best.pth \
  --max-items 2 \
  --batch-size 1
'''
