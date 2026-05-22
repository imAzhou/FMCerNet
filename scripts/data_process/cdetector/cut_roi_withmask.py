import argparse
import json
import os
from collections import defaultdict
from multiprocessing import Pool

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils
from scipy import sparse
from scipy.ndimage import find_objects
from tqdm import tqdm


DATASET_ROOT = "/shared_storage/xzly/datasets/CervicalDatasets/ComparisonDetectorDataset"
OUTPUT_ROOT = "data_resource/CDetector_WS400"
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
    parser = argparse.ArgumentParser(description="Cut CDetector images with SAM2 masks.")
    parser.add_argument("--dataset-root", default=DATASET_ROOT)
    parser.add_argument("--output-root", default=OUTPUT_ROOT)
    parser.add_argument("--modes", nargs="+", default=["train", "test"], choices=["train", "test"])
    parser.add_argument("--window-size", type=int, default=WINDOW_SIZE)
    parser.add_argument("--stride", type=int, default=STRIDE)
    parser.add_argument("--minlen", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None, help="Optional smoke-test limit per mode.")
    return parser.parse_args()


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
    ann_category_ids = []
    ann_masks = []
    annidx = np.unique(patch_mask)
    if len(annidx) <= 1:
        return ann_bboxes, ann_category_ids, ann_masks

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

        annitem = annlist[aidx - 1]
        inst_mask = (patch_mask == aidx)
        ann_bboxes.append([bx1, by1, bx2, by2])
        ann_category_ids.append(POSITIVE_CLASSES.index(annitem["mapped_name"]) + 1)
        ann_masks.append(inst_mask)

    return ann_bboxes, ann_category_ids, ann_masks


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


def process_items(worker_args):
    proc_id, args_dict, items = worker_args
    dataset_root = args_dict["dataset_root"]
    output_root = args_dict["output_root"]
    window_size = args_dict["window_size"]
    stride = args_dict["stride"]
    minlen = args_dict["minlen"]
    img_save_dir = os.path.join(output_root, f"WINDOW_SIZE_{window_size}", "images")
    mask_root = os.path.join(output_root, "roi_inst_mask")

    patch_items = []
    for item in tqdm(items, ncols=80, position=proc_id, desc=f"worker {proc_id}"):
        img_path = os.path.join(dataset_root, item["mode"], item["file_name"])
        roi_img = Image.open(img_path).convert("RGB")
        rw, rh = roi_img.size
        purename = os.path.splitext(item["file_name"])[0]

        if item["annos"]:
            mask_path = os.path.join(mask_root, item["mode"], f"{purename}.npz")
            if not os.path.exists(mask_path):
                raise FileNotFoundError(f"Missing mask file: {mask_path}")
            roi_mask = load_sparse_mask(mask_path)
        else:
            roi_mask = np.zeros((rh, rw), dtype=np.int16)

        cut_points = generate_cut_regions((0, 0), rw, rh, window_size, stride, minlen=minlen)
        for patch_idx, patch_coords in enumerate(cut_points):
            bboxes, category_ids, patch_masks = calc_patch_anns(
                patch_coords, item["annos"], roi_mask
            )
            diagnose = int(len(bboxes) > 0)
            prefix = "Pos" if diagnose else "Neg"
            filename = f"{purename}_{patch_idx}.png"
            cropimg = cut_img(roi_img, patch_coords, window_size)
            cropimg.save(os.path.join(img_save_dir, prefix, filename))

            instances = []
            for bbox, category_id, annmask in zip(bboxes, category_ids, patch_masks):
                bx1, by1, bx2, by2 = bbox
                rle = mask_utils.encode(np.asfortranarray(annmask.astype(np.uint8)))
                rle["counts"] = rle["counts"].decode("utf-8")
                instances.append({
                    "bbox": [bx1, by1, bx2 - bx1, by2 - by1],
                    "bbox_xyxy": bbox,
                    "label": category_id - 1,
                    "category_id": category_id,
                    "category_name": POSITIVE_CLASSES[category_id - 1],
                    "mask": rle,
                    "area": int(annmask.sum()),
                })

            patch_items.append({
                "file_name": f"{prefix}/{filename}",
                "gt_label": sorted({category_id - 1 for category_id in category_ids}),
                "instances": instances,
                "extra_info": {
                    "mode": item["mode"],
                    "source_file": item["file_name"],
                    "square_coords": patch_coords,
                },
                "diagnose": diagnose,
            })

    return patch_items


def run_mode(args, mode):
    img_save_dir = os.path.join(args.output_root, f"WINDOW_SIZE_{args.window_size}", "images")
    ann_save_dir = os.path.join(args.output_root, f"WINDOW_SIZE_{args.window_size}", "annofiles")
    os.makedirs(os.path.join(img_save_dir, "Neg"), exist_ok=True, mode=0o777)
    os.makedirs(os.path.join(img_save_dir, "Pos"), exist_ok=True, mode=0o777)
    os.makedirs(ann_save_dir, exist_ok=True, mode=0o777)

    items = load_mode_items(args.dataset_root, mode, args.limit)
    worker_num = min(args.num_workers, max(1, len(items)))
    splits = np.array_split(np.arange(len(items)), worker_num)
    args_dict = vars(args)
    task_args = [
        (proc_id, args_dict, [items[i] for i in split.tolist()])
        for proc_id, split in enumerate(splits)
        if len(split) > 0
    ]

    if len(task_args) == 1:
        worker_results = [process_items(task_args[0])]
    else:
        with Pool(processes=len(task_args)) as pool:
            worker_results = pool.map(process_items, task_args)

    patch_items = []
    for result in worker_results:
        patch_items.extend(result)

    save_path = os.path.join(ann_save_dir, f"patches_{mode}.json")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(patch_items, f, ensure_ascii=False)

    pn_cnt = [0, 0]
    for item in patch_items:
        pn_cnt[item["diagnose"]] += 1
    print(f"{mode}: patches={len(patch_items)}, pn_cnt={pn_cnt}")
    print(f"Saved: {save_path}")


def main():
    args = parse_args()
    for mode in args.modes:
        run_mode(args, mode)


if __name__ == "__main__":
    main()
