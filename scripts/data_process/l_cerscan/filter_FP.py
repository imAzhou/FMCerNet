import argparse
import json
import os
from pathlib import Path

from mmengine.fileio import load
from tqdm import tqdm


PRED_RESULT = "work_dir/mlc/smartccs/fb_our_decoder/neg1750/pred_result.pkl"
DATA_ROOT = "data_resource/LCerScan/WS800"
TEMPLATE_ANN_FILE = "annofiles/multilabel_puretrain.json"
OUTPUT_FILE = "work_dir/mlc/smartccs/fb_our_decoder/neg1750/filter_FP.json"
THRESHOLD = 0.5


def parse_args():
    parser = argparse.ArgumentParser(
        description="Filter false-positive LCerScan patches from binary prediction results."
    )
    parser.add_argument("--pred-result", default=PRED_RESULT)
    parser.add_argument("--data-root", default=DATA_ROOT)
    parser.add_argument("--template-ann-file", default=TEMPLATE_ANN_FILE)
    parser.add_argument("--output-file", default=OUTPUT_FILE)
    parser.add_argument("--threshold", type=float, default=THRESHOLD)
    return parser.parse_args()


def resolve_path(data_root, path):
    path = Path(path)
    if path.is_absolute() or path.is_file():
        return path
    return Path(data_root) / path


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
    img_path = os.path.abspath(os.path.normpath(img_path))
    image_root = os.path.abspath(os.path.normpath(image_root))
    if os.path.commonpath([img_path, image_root]) != image_root:
        raise ValueError(f"{img_path} is not under {image_root}")
    return os.path.relpath(img_path, image_root).replace(os.sep, "/")


def load_template_metainfo(data_root, template_ann_file):
    template_path = resolve_path(data_root, template_ann_file)
    with template_path.open("r", encoding="utf-8") as f:
        template_ann = json.load(f)
    if "metainfo" not in template_ann or "data_list" not in template_ann:
        raise KeyError(f"{template_path} must contain metainfo and data_list")
    return template_ann["metainfo"]


def filter_false_positives(pred_result, data_root, threshold):
    image_root = Path(data_root) / "images"
    if not image_root.is_dir():
        raise FileNotFoundError(f"Image root not found: {image_root}")

    results = load(pred_result)
    fp_items = []
    seen_img_paths = set()
    for item in tqdm(results, ncols=80):
        if not hasattr(item, "img_path"):
            raise AttributeError("Prediction item missing img_path")
        if not hasattr(item, "gt_label"):
            raise AttributeError("Prediction item missing gt_label")
        if not hasattr(item, "img_prob"):
            raise AttributeError("Prediction item missing img_prob")

        gt_label = tensor_to_list(item.gt_label)
        img_prob = scalar_to_float(item.img_prob)
        if gt_label != [] or img_prob <= threshold:
            continue

        img_path = rel_img_path(item.img_path, image_root)
        if img_path in seen_img_paths:
            raise ValueError(f"Duplicate FP image path: {img_path}")
        seen_img_paths.add(img_path)
        fp_items.append(
            {
                "img_path": img_path,
                "gt_label": [],
            }
        )
    return fp_items, len(results)


def dump_json(data, output_file):
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    return output_path


def main():
    args = parse_args()
    metainfo = load_template_metainfo(args.data_root, args.template_ann_file)
    fp_items, total_count = filter_false_positives(args.pred_result, args.data_root, args.threshold)
    output_ann = {
        "metainfo": metainfo,
        "data_list": fp_items,
    }
    output_path = dump_json(output_ann, args.output_file)
    print(
        f"Saved {output_path}: {len(fp_items)} FP samples "
        f"from {total_count} predictions with img_prob > {args.threshold}"
    )


if __name__ == "__main__":
    main()

