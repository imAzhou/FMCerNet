import argparse
import json
import os
from collections import defaultdict

from prettytable import PrettyTable
from tqdm import tqdm


OUTPUT_ROOT = "data_resource/CDetector_WS400"
WINDOW_SIZE = 400
POSITIVE_CLASSES = ["AGC", "ASC-US", "LSIL", "ASC-H", "HSIL"]


def parse_args():
    parser = argparse.ArgumentParser(description="Build CDetector multilabel JSON files.")
    parser.add_argument("--output-root", default=OUTPUT_ROOT)
    parser.add_argument("--window-size", type=int, default=WINDOW_SIZE)
    parser.add_argument("--modes", nargs="+", default=["train", "test"], choices=["train", "test"])
    return parser.parse_args()


def build_multilabel(args, mode):
    ann_dir = os.path.join(args.output_root, f"WINDOW_SIZE_{args.window_size}", "annofiles")
    patch_json_path = os.path.join(ann_dir, f"patches_{mode}.json")
    with open(patch_json_path, "r", encoding="utf-8") as f:
        patch_items = json.load(f)

    multilabel_data = {
        "metainfo": {"classes": POSITIVE_CLASSES},
        "data_list": [],
    }
    pn_cnt = [0, 0]
    for item in tqdm(patch_items, ncols=80, desc=mode):
        gt_label = sorted(set(item.get("gt_label", [])))
        multilabel_data["data_list"].append({
            "img_path": item["file_name"],
            "gt_label": gt_label,
        })
        pn_cnt[int(len(gt_label) > 0)] += 1

    save_path = os.path.join(ann_dir, f"multilabel_{mode}.json")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(multilabel_data, f, ensure_ascii=False)

    print(f"{mode} pn_cnt: {pn_cnt}")
    print(f"Saved: {save_path}")
    return save_path


def statistic(json_paths):
    for mode, json_path in json_paths:
        with open(json_path, "r", encoding="utf-8") as f:
            json_data = json.load(f)
        label_count = defaultdict(int)
        for item in json_data["data_list"]:
            for label in item.get("gt_label", []):
                label_count[label] += 1

        table = PrettyTable(title=mode)
        table.field_names = POSITIVE_CLASSES
        table.add_row([label_count.get(i, 0) for i in range(len(POSITIVE_CLASSES))])
        print(table)


def main():
    args = parse_args()
    json_paths = []
    for mode in args.modes:
        json_paths.append((mode, build_multilabel(args, mode)))
    statistic(json_paths)


if __name__ == "__main__":
    main()
