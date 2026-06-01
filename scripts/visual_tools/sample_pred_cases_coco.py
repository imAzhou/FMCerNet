import argparse
import copy
import json
import pickle
import random
from collections import defaultdict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sample TP/FP/TN/FN cases from prediction pkl into COCO json."
    )
    parser.add_argument("pred_pkl", type=Path)
    parser.add_argument("coco_json", type=Path)
    parser.add_argument("out_json", type=Path)
    parser.add_argument("--per-type", type=int, default=100)
    parser.add_argument("--thr", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def load_json(path):
    with path.open("r") as f:
        return json.load(f)


def dump_json(data, path):
    if str(path).startswith("/tmp"):
        raise ValueError("Temporary outputs must be saved under work_dir/tmp, not /tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f, ensure_ascii=False)


def pred_file_name(img_path):
    path = Path(img_path)
    parts = path.parts
    if "images" not in parts:
        raise ValueError(f"Prediction img_path must contain an images directory: {img_path}")
    images_idx = len(parts) - 1 - parts[::-1].index("images")
    return Path(*parts[images_idx + 1:]).as_posix()


def tensor_to_float(value):
    return float(value.detach().cpu().item())


def tensor_to_list(value):
    return [float(item) for item in value.detach().cpu().tolist()]


def case_type(is_positive, is_pred_positive):
    if is_positive and is_pred_positive:
        return "TP"
    if not is_positive and is_pred_positive:
        return "FP"
    if not is_positive and not is_pred_positive:
        return "TN"
    if is_positive and not is_pred_positive:
        return "FN"
    raise ValueError("Unreachable case type.")


def build_pred_records(pred_pkl, category_ids, thr):
    with pred_pkl.open("rb") as f:
        samples = pickle.load(f)

    records = []
    for sample in samples:
        pos_prob = tensor_to_list(sample.pos_prob)
        if len(pos_prob) != len(category_ids):
            raise ValueError(
                f"pos_prob length must be {len(category_ids)}, got {len(pos_prob)}."
            )

        img_prob = tensor_to_float(sample.img_prob)
        is_positive = len(sample.gt_label) > 0
        is_pred_positive = img_prob >= thr
        records.append(
            {
                "file_name": pred_file_name(sample.img_path),
                "sample_idx": int(sample.sample_idx),
                "width": int(sample.ori_shape[1]),
                "height": int(sample.ori_shape[0]),
                "case_type": case_type(is_positive, is_pred_positive),
                "img_prob": img_prob,
                "pos_prob": pos_prob,
                "img_pred_pn": int(is_pred_positive),
                "img_pred_mcls": [
                    category_id
                    for category_id, prob in zip(category_ids, pos_prob)
                    if prob >= thr
                ],
            }
        )
    return records


def sample_records(records, per_type, seed):
    grouped = defaultdict(list)
    for record in records:
        grouped[record["case_type"]].append(record)

    rng = random.Random(seed)
    sampled = []
    for name in ["TP", "FP", "TN", "FN"]:
        candidates = grouped[name]
        sample_num = min(per_type, len(candidates))
        sampled.extend(rng.sample(candidates, sample_num))
    return sampled


def build_coco_subset(coco, sampled_records):
    category_ids = [item["id"] for item in coco["categories"]]
    if category_ids != sorted(category_ids):
        raise ValueError(f"Category ids must be sorted, got {category_ids}.")

    images_by_name = {item["file_name"]: item for item in coco["images"]}
    anns_by_image = defaultdict(list)
    for ann in coco["annotations"]:
        anns_by_image[ann["image_id"]].append(ann)

    out_images = []
    out_annotations = []
    ann_id = 0
    for image_id, record in enumerate(sampled_records):
        if record["file_name"] in images_by_name:
            image_info = copy.deepcopy(images_by_name[record["file_name"]])
        else:
            image_info = {
                "width": record["width"],
                "height": record["height"],
                "file_name": record["file_name"],
            }

        old_image_id = image_info.get("id", None)
        image_info["id"] = image_id
        for key in [
            "sample_idx",
            "case_type",
            "img_prob",
            "pos_prob",
            "img_pred_pn",
            "img_pred_mcls",
        ]:
            image_info[key] = record[key]
        out_images.append(image_info)

        if old_image_id is None:
            continue

        for ann in anns_by_image[old_image_id]:
            out_ann = copy.deepcopy(ann)
            out_ann["id"] = ann_id
            out_ann["image_id"] = image_id
            out_annotations.append(out_ann)
            ann_id += 1

    return {
        "categories": coco["categories"],
        "images": out_images,
        "annotations": out_annotations,
        "info": coco.get("info", []),
    }


def main():
    args = parse_args()
    coco = load_json(args.coco_json)
    for key in ["categories", "images", "annotations"]:
        if key not in coco:
            raise KeyError(f"COCO json must contain key: {key}")

    category_ids = [item["id"] for item in coco["categories"]]
    records = build_pred_records(args.pred_pkl, category_ids, args.thr)
    sampled_records = sample_records(records, args.per_type, args.seed)
    out_coco = build_coco_subset(coco, sampled_records)
    dump_json(out_coco, args.out_json)

    counts = defaultdict(int)
    for image in out_coco["images"]:
        counts[image["case_type"]] += 1
    print(
        f"Saved {len(out_coco['images'])} images and "
        f"{len(out_coco['annotations'])} annotations to {args.out_json}"
    )
    print({name: counts[name] for name in ["TP", "FP", "TN", "FN"]})


if __name__ == "__main__":
    main()
