import argparse
import json
import math
import os
import warnings
from pathlib import Path
from types import SimpleNamespace

os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import mmengine.dist as dist
import numpy as np
import torch
from PIL import Image, ImageDraw
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from fmcernet.nets.backbone.SmartCCS_backbone import SmartCCS
from fmcernet.utils import init_distributed_mode, is_main_process, set_seed


DATA_ROOT = "data_resource/LCerScan/WS800"
CANDIDATE_ANN_FILE = "annofiles/multilabel_puretrain.json"
TOTAL_ANN_FILE = "annofiles/multilabel_puretrain.json"
CKPT = "checkpoints/CCS_vitl_100M.pth"
OUTPUT_FILE = "data_resource/LCerScan/WS800/annofiles/multilabel_puretrain_neg1000.json"
COLLECT_TMPDIR = "work_dir/tmp/l_cerscan_neg_mining_collect"
VIS_OUTPUT_FILE = "work_dir/tmp/l_cerscan_neg_mining_clusters.jpg"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Mine 1000 LCerScan WS800 training negatives with SmartCCS DDP features."
    )
    parser.add_argument("--data-root", default=DATA_ROOT)
    parser.add_argument("--candidate-ann-file", default=CANDIDATE_ANN_FILE)
    parser.add_argument("--total-ann-file", default=TOTAL_ANN_FILE)
    parser.add_argument("--current-ann-file", default=None)
    parser.add_argument("--ckpt", default=CKPT)
    parser.add_argument("--output-file", default=OUTPUT_FILE)
    parser.add_argument("--vis-output-file", default=VIS_OUTPUT_FILE)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--num-clusters", type=int, default=200)
    parser.add_argument("--samples-per-cluster", type=int, default=5)
    parser.add_argument("--pca-dim", type=int, default=256)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--world-size", default=3, type=int)
    parser.add_argument("--dist-url", default="env://")
    return parser.parse_args()


class NegativeImageDataset(Dataset):
    def __init__(self, image_root, negative_items):
        self.image_root = Path(image_root)
        self.negative_items = negative_items
        self.pixel_mean = torch.tensor([123.675, 116.28, 103.53], dtype=torch.float32).view(3, 1, 1)
        self.pixel_std = torch.tensor([58.395, 57.12, 57.375], dtype=torch.float32).view(3, 1, 1)

    def __len__(self):
        return len(self.negative_items)

    def __getitem__(self, index):
        item = self.negative_items[index]
        img_path = self.image_root / item["img_path"]
        image = Image.open(img_path).convert("RGB")
        image = image.resize((224, 224), resample=Image.Resampling.BICUBIC)
        image = torch.from_numpy(np.asarray(image, dtype=np.float32)).permute(2, 0, 1)
        image = (image - self.pixel_mean) / self.pixel_std
        return {
            "index": index,
            "image": image,
        }


class DDPFeatureModel(torch.nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.ddp_anchor = torch.nn.Parameter(torch.zeros(()))

    def forward(self, images):
        return self.backbone(images)


def resolve_ann_path(data_root, ann_file):
    ann_path = Path(ann_file)
    if ann_path.is_absolute() or ann_path.is_file():
        return ann_path
    ann_path = Path(data_root) / ann_path
    return ann_path


def load_ann_data(data_root, ann_file):
    data_root = Path(data_root)
    image_root = data_root / "images"
    ann_path = resolve_ann_path(data_root, ann_file)

    if not ann_path.is_file():
        raise FileNotFoundError(f"Annotation file not found: {ann_path}")
    if not image_root.is_dir():
        raise FileNotFoundError(f"Image root not found: {image_root}")

    with ann_path.open("r", encoding="utf-8") as f:
        ann_data = json.load(f)

    if "metainfo" not in ann_data or "data_list" not in ann_data:
        raise KeyError(f"{ann_path} must contain metainfo and data_list")
    if not isinstance(ann_data["data_list"], list):
        raise TypeError(f"{ann_path} data_list must be a list")

    for data_index, item in enumerate(ann_data["data_list"]):
        if "img_path" not in item:
            raise KeyError(f"Missing img_path at data_list[{data_index}]")
        if "gt_label" not in item:
            raise KeyError(f"Missing gt_label at data_list[{data_index}]")
        if not isinstance(item["gt_label"], list):
            raise TypeError(f"gt_label must be a list at data_list[{data_index}]")

        img_path = image_root / item["img_path"]
        if not img_path.is_file():
            raise FileNotFoundError(f"Image file not found: {img_path}")

    return ann_data, image_root


def split_positive_negative_items(ann_data):
    positive_items = []
    negative_items = []
    for data_index, item in enumerate(ann_data["data_list"]):
        if item["gt_label"] == []:
            negative_items.append(
                {
                    "data_index": data_index,
                    "img_path": item["img_path"],
                    "ann_item": item,
                }
            )
        else:
            positive_items.append(item)

    return positive_items, negative_items


def load_mining_inputs(data_root, candidate_ann_file, total_ann_file, current_ann_file, limit):
    candidate_ann_data, image_root = load_ann_data(data_root, candidate_ann_file)
    candidate_positive_items, candidate_negative_items = split_positive_negative_items(candidate_ann_data)
    raw_candidate_negative_count = len(candidate_negative_items)
    candidate_negative_paths = {item["img_path"] for item in candidate_negative_items}

    total_ann_data, _ = load_ann_data(data_root, total_ann_file)
    base_positive_items, total_negative_items = split_positive_negative_items(total_ann_data)

    if current_ann_file is None:
        base_ann_data = total_ann_data
        base_data_list = base_positive_items
        current_negative_paths = set()
    else:
        current_ann_data, _ = load_ann_data(data_root, current_ann_file)
        current_positive_items, current_negative_items = split_positive_negative_items(current_ann_data)
        if current_positive_items and is_main_process():
            print(f"Ignore {len(current_positive_items)} positives in current-ann-file")
        base_ann_data = total_ann_data
        base_data_list = list(base_positive_items)
        base_data_list.extend(item["ann_item"] for item in current_negative_items)
        current_negative_paths = {item["img_path"] for item in current_negative_items}

    candidate_negative_items = [
        item
        for item in candidate_negative_items
        if item["img_path"] not in current_negative_paths
    ]
    random_negative_items = [
        item
        for item in total_negative_items
        if item["img_path"] not in candidate_negative_paths
        and item["img_path"] not in current_negative_paths
    ]

    if limit is not None:
        if limit <= 0:
            raise ValueError(f"--limit must be positive, got {limit}")
        candidate_negative_items = candidate_negative_items[:limit]

    return (
        base_ann_data,
        base_data_list,
        image_root,
        candidate_negative_items,
        len(candidate_positive_items),
        raw_candidate_negative_count,
        len(candidate_negative_items),
        len(current_negative_paths),
        len(total_negative_items),
        len(random_negative_items),
        random_negative_items,
    )


def build_smartccs(ckpt):
    args = SimpleNamespace(
        backbone_cfg={
            "backbone_ckpt": ckpt,
            "frozen_backbone": True,
            "use_peft": None,
        }
    )
    return SmartCCS(args)


def extract_ddp_features(model, dataloader, device, total_size, collect_tmpdir):
    model.eval()
    batch_results = []
    pbar = dataloader
    if is_main_process():
        pbar = tqdm(dataloader, ncols=80)

    with torch.inference_mode():
        for batch in pbar:
            images = batch["image"].to(device, non_blocking=True)
            indices = batch["index"].cpu().numpy().tolist()
            outputs = model(images)
            if "x_norm_clstoken" not in outputs:
                raise KeyError("SmartCCS output missing x_norm_clstoken")
            features = outputs["x_norm_clstoken"].detach().cpu().numpy()
            for sample_index, feature in zip(indices, features):
                batch_results.append((int(sample_index), feature.astype(np.float32)))

    if is_main_process():
        pbar.close()

    return dist.collect_results(batch_results, total_size, device="cpu", tmpdir=collect_tmpdir)


def select_negatives(collected_results, negative_items, num_clusters, samples_per_cluster, pca_dim, seed):
    if collected_results is None:
        return None, None
    if len(collected_results) != len(negative_items):
        raise ValueError(f"Expected {len(negative_items)} features, got {len(collected_results)}")
    print(f"Collected {len(collected_results)} features")

    collected_results = sorted(collected_results, key=lambda item: item[0])
    indices = [item[0] for item in collected_results]
    expected_indices = list(range(len(negative_items)))
    if indices != expected_indices:
        raise ValueError("Collected feature indices do not match negative sample order")

    features = np.stack([item[1] for item in collected_results], axis=0)
    if pca_dim <= 0:
        raise ValueError(f"--pca-dim must be positive, got {pca_dim}")
    max_pca_dim = min(features.shape)
    if pca_dim > max_pca_dim:
        raise ValueError(f"--pca-dim must be <= {max_pca_dim}, got {pca_dim}")
    pca = PCA(n_components=pca_dim, random_state=seed)
    features = pca.fit_transform(features)
    print(f"PCA finished: reduced_features={features.shape}")

    print(f"Start KMeans: features={features.shape}, num_clusters={num_clusters}")
    kmeans = KMeans(n_clusters=num_clusters, random_state=seed, n_init=10)
    cluster_ids = kmeans.fit_predict(features)
    print("KMeans finished")

    rng = np.random.default_rng(seed)
    target_negative_count = num_clusters * samples_per_cluster
    if len(negative_items) < target_negative_count:
        raise ValueError(
            f"Need {target_negative_count} negatives, got {len(negative_items)} available"
        )

    selected_negative_indices = []
    selected_negative_set = set()
    small_cluster_count = 0
    small_cluster_selected_count = 0
    for cluster_id in range(num_clusters):
        member_indices = np.flatnonzero(cluster_ids == cluster_id)
        if len(member_indices) < samples_per_cluster:
            selected = member_indices
            small_cluster_count += 1
            small_cluster_selected_count += len(member_indices)
        else:
            selected = rng.choice(member_indices, size=samples_per_cluster, replace=False)
        for index in selected:
            selected_index = int(index)
            selected_negative_indices.append(selected_index)
            selected_negative_set.add(selected_index)

    remaining_needed = target_negative_count - len(selected_negative_indices)
    if remaining_needed > 0:
        remaining_indices = [
            index
            for index in range(len(negative_items))
            if index not in selected_negative_set
        ]
        if len(remaining_indices) < remaining_needed:
            raise ValueError(
                f"Need {remaining_needed} extra negatives, got {len(remaining_indices)} available"
            )
        extra_selected = rng.choice(remaining_indices, size=remaining_needed, replace=False)
        selected_negative_indices.extend(int(index) for index in extra_selected)
        print(
            f"{small_cluster_count} clusters had fewer than {samples_per_cluster} samples; "
            f"selected all {small_cluster_selected_count} samples from them and "
            f"randomly filled {remaining_needed} extras from other clusters"
        )

    if len(selected_negative_indices) != target_negative_count:
        raise ValueError(
            f"Expected {target_negative_count} selected negatives, "
            f"got {len(selected_negative_indices)}"
        )
    if len(set(selected_negative_indices)) != len(selected_negative_indices):
        raise ValueError("Selected negative indices are not unique")
    print(f"Selected {len(selected_negative_indices)} negatives")

    return selected_negative_indices, cluster_ids


def save_cluster_visual(image_root, negative_items, cluster_ids, num_clusters, seed, output_file):
    rng = np.random.default_rng(seed)
    grid_size = math.ceil(math.sqrt(num_clusters))
    thumb_size = 224
    canvas = Image.new("RGB", (grid_size * thumb_size, grid_size * thumb_size), color=(255, 255, 255))

    for cluster_id in range(num_clusters):
        member_indices = np.flatnonzero(cluster_ids == cluster_id)
        if len(member_indices) == 0:
            raise ValueError(f"Cluster {cluster_id} has no samples for visualization")
        negative_index = int(rng.choice(member_indices))
        img_path = Path(image_root) / negative_items[negative_index]["img_path"]
        image = Image.open(img_path).convert("RGB")
        image = image.resize((thumb_size, thumb_size), resample=Image.Resampling.BICUBIC)
        row = cluster_id // grid_size
        col = cluster_id % grid_size
        canvas.paste(image, (col * thumb_size, row * thumb_size))

    draw = ImageDraw.Draw(canvas)
    line_color = (255, 0, 0)
    line_width = 3
    for grid_index in range(grid_size + 1):
        offset = grid_index * thumb_size
        draw.line([(offset, 0), (offset, canvas.height)], fill=line_color, width=line_width)
        draw.line([(0, offset), (canvas.width, offset)], fill=line_color, width=line_width)

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=95)
    return output_path


def select_random_negatives(random_negative_items, random_count, seed):
    if random_count <= 0:
        return []
    if len(random_negative_items) < random_count:
        raise ValueError(
            f"Need {random_count} random negatives, got {len(random_negative_items)} available"
        )
    rng = np.random.default_rng(seed)
    selected = rng.choice(len(random_negative_items), size=random_count, replace=False)
    return [int(index) for index in selected]


def get_random_negative_count(args, selected_negative_count, random_negative_items):
    if args.current_ann_file is None and args.candidate_ann_file == args.total_ann_file:
        return 0
    return selected_negative_count // 2


def build_output_ann(
    base_ann_data,
    base_data_list,
    negative_items,
    selected_negative_indices,
    random_negative_items,
    selected_random_negative_indices,
):
    selected_negative_items = [
        negative_items[negative_index]
        for negative_index in selected_negative_indices
    ]
    selected_random_negative_items = [
        random_negative_items[negative_index]
        for negative_index in selected_random_negative_indices
    ]

    expected_negative_count = len(selected_negative_indices)
    if len(selected_negative_items) != expected_negative_count:
        raise ValueError(
            f"Expected {expected_negative_count} selected negatives, got {len(selected_negative_items)}"
        )

    output_data_list = list(base_data_list)
    output_data_list.extend(item["ann_item"] for item in selected_negative_items)
    output_data_list.extend(item["ann_item"] for item in selected_random_negative_items)
    output_paths = [item["img_path"] for item in output_data_list]
    if len(output_paths) != len(set(output_paths)):
        raise ValueError("Output data_list contains duplicate img_path values")

    positive_count = sum(1 for item in output_data_list if item["gt_label"] != [])
    negative_count = sum(1 for item in output_data_list if item["gt_label"] == [])

    return {
        "metainfo": base_ann_data["metainfo"],
        "data_list": output_data_list,
    }, positive_count, negative_count


def dump_json(data, data_root, output_file):
    output_path = Path(output_file)
    if not output_path.is_absolute():
        output_path = Path(data_root) / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    return output_path


def main():
    args = parse_args()
    init_distributed_mode(args)
    if not getattr(args, "distributed", False):
        raise RuntimeError("neg_mining.py must be launched with torchrun/DDP")

    set_seed(args.seed)
    device = torch.device(f"cuda:{os.getenv('LOCAL_RANK')}")

    (
        base_ann_data,
        base_data_list,
        image_root,
        negative_items,
        candidate_positive_count,
        raw_candidate_negative_count,
        candidate_negative_count,
        current_negative_count,
        total_negative_count,
        random_negative_count,
        random_negative_items,
    ) = load_mining_inputs(
        args.data_root,
        args.candidate_ann_file,
        args.total_ann_file,
        args.current_ann_file,
        args.limit,
    )
    if is_main_process():
        print(
            f"Loaded {candidate_positive_count} candidate positives, "
            f"{raw_candidate_negative_count} raw candidate negatives, "
            f"{candidate_negative_count} candidate negatives after filtering, "
            f"{total_negative_count} total negatives, "
            f"{random_negative_count} random negative candidates, "
            f"{current_negative_count} current negatives, "
            f"{len(base_data_list)} base samples"
        )

    dataset = NegativeImageDataset(image_root, negative_items)
    sampler = DistributedSampler(dataset, shuffle=False)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    try:
        model = DDPFeatureModel(build_smartccs(args.ckpt)).to(device)
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.gpu],
            find_unused_parameters=True,
        )

        collected_results = extract_ddp_features(
            model,
            dataloader,
            device,
            len(dataset),
            COLLECT_TMPDIR,
        )
        if is_main_process():
            print("Feature extraction finished")
        selected_negative_indices, cluster_ids = select_negatives(
            collected_results,
            negative_items,
            args.num_clusters,
            args.samples_per_cluster,
            args.pca_dim,
            args.seed,
        )

        if is_main_process():
            if args.visualize:
                visual_path = save_cluster_visual(
                    image_root,
                    negative_items,
                    cluster_ids,
                    args.num_clusters,
                    args.seed,
                    args.vis_output_file,
                )
                print(f"Saved cluster visualization: {visual_path}")
            random_count = get_random_negative_count(
                args,
                len(selected_negative_indices),
                random_negative_items,
            )
            selected_random_negative_indices = select_random_negatives(
                random_negative_items,
                random_count,
                args.seed,
            )
            print(
                f"Selected {len(selected_negative_indices)} clustered hard negatives "
                f"and {len(selected_random_negative_indices)} random negatives"
            )
            output_ann, output_positive_count, output_negative_count = build_output_ann(
                base_ann_data,
                base_data_list,
                negative_items,
                selected_negative_indices,
                random_negative_items,
                selected_random_negative_indices,
            )
            output_path = dump_json(output_ann, args.data_root, args.output_file)
            print(
                f"Saved {output_path}: "
                f"{output_positive_count} positives, {output_negative_count} negatives, "
                f"{len(output_ann['data_list'])} total"
            )
    finally:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()


'''
Initialize samples from the full puretrain json:

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 --master_port=12349 scripts/data_process/l_cerscan/neg_mining.py \
    --candidate-ann-file annofiles/multilabel_puretrain.json \
    --total-ann-file annofiles/multilabel_puretrain.json \
    --output-file annofiles/multilabel_puretrain_neg3000.json \
    --num-clusters 300 \
    --samples-per-cluster 10 \
    --visualize

Append samples from a negative-only candidate json:

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 --master_port=12349 scripts/data_process/l_cerscan/neg_mining.py \
    --candidate-ann-file work_dir/mlc/smartccs/fb_our_decoder/neg1750/filter_FP.json \
    --total-ann-file annofiles/multilabel_puretrain.json \
    --current-ann-file annofiles/multilabel_puretrain_neg1750_mlc.json \
    --output-file annofiles/multilabel_puretrain_neg2500_mlc.json \
    --num-clusters 50 \
    --samples-per-cluster 10 \
    --visualize \
    --vis-output-file work_dir/mlc/smartccs/fb_our_decoder/neg1750/clusters_50.jpg
'''
