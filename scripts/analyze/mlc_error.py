import torch
import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=r"^xFormers is available \(.*\)")
from tqdm import tqdm
import argparse
from mmengine.config import Config
import mmengine.dist as dist
from mmengine.fileio import dump
import pickle
from pycocotools.coco import COCO
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from collections import Counter
from prettytable import PrettyTable
from cerwsi.datasets import load_data
from cerwsi.nets import PatchNet
from cerwsi.utils import set_seed, init_distributed_mode, is_main_process


parser = argparse.ArgumentParser()
# base args
parser.add_argument('config_file', type=str)
parser.add_argument('ckpt', type=str)
parser.add_argument('save_dir', type=str)
parser.add_argument('--save_result', action='store_true')
parser.add_argument('--visual_FN_flag', action='store_true')
parser.add_argument('--visual_FN_cocojsonfile', type=str, help='use to visual gt bboxes')
parser.add_argument('--visual_FN_nums', type=int, default=-1, help='-1 means visual all FN imgs')
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--print_interval', type=int, default=10, help='random seed')
parser.add_argument('--world_size', default=3, type=int, help='number of distributed processes')
parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

args = parser.parse_args()

def test_net(args, cfg, model):
    valloader = load_data(cfg, ['val'])

    model.eval()
    pbar = valloader
    if is_main_process():
        pbar = tqdm(valloader, ncols=80)
    
    batch_outputs = []
    for idx, data_batch in enumerate(pbar):
        # if idx > 2:
        #     break
        with torch.no_grad():
            outputs = model(data_batch, 'val')
        batch_outputs.extend([item.cpu() for item in outputs])

    results = dist.collect_results(batch_outputs, len(valloader.dataset), device='cpu')
    if is_main_process():
        pbar.close()
        save_path = f"{args.save_dir}/error_analyze.txt"
        analyze(results, cfg.classes, save_path)

        if args.save_result:
            out_file_path = f'{args.save_dir}/pred_result.pkl'
            dump(results, out_file_path)
            print(f'Results saved in {out_file_path}')
        
        if args.visual_FN_flag:
            img_savedir = f'{args.save_dir}/visual_FN'
            os.makedirs(img_savedir, exist_ok=True, mode=0o777)
            visual_FN(results, args.visual_FN_cocojsonfile, args.visual_FN_nums, img_savedir)

def main():
    init_distributed_mode(args)
    set_seed(args.seed)
    device = torch.device(f'cuda:{os.getenv("LOCAL_RANK")}')

    cfg = Config.fromfile(args.config_file)
    cfg.backbone_cfg['backbone_ckpt'] = None
    cfg.instance_ckpt = None
    model = PatchNet(cfg).to(device)
    model_without_ddp = model

    os.makedirs(args.save_dir, exist_ok=True, mode=0o777)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    
    model_without_ddp.load_ckpt(args.ckpt)
    test_net(args, cfg, model)

    # if args.distributed:
    #     dist.barrier()
    #     dist.destroy_process_group()

def analyze(pred_result, classes, save_path):
    thr = 0.5
    missed_poscls, FP_poscls, total_poscls = [], [], []
    matrix = {
        "pred_diag_0": {"pred_multilabel_0": 0, "pred_multilabel_1": 0},
        "pred_diag_1": {"pred_multilabel_0": 0, "pred_multilabel_1": 0}
    }

    for item in tqdm(pred_result, ncols=80):
        gt_diagnose = int(len(item.gt_label)>0)
        pred_diagnose = int(item.img_prob > 0.5)
        gt_multi_label = item.gt_label.tolist()
        pred_multi_label = [clsidx for clsidx,cls_score in enumerate(item.pos_prob) if cls_score > thr]
        total_poscls.extend(gt_multi_label)
        
        # 更新一致性矩阵
        if pred_diagnose == 0:
            if len(pred_multi_label) == 0:
                matrix["pred_diag_0"]["pred_multilabel_0"] += 1
            else:
                matrix["pred_diag_0"]["pred_multilabel_1"] += 1
        else:  # pred_diagnose == 1
            if len(pred_multi_label) == 0:
                matrix["pred_diag_1"]["pred_multilabel_0"] += 1
            else:
                matrix["pred_diag_1"]["pred_multilabel_1"] += 1

        # 错误情况统计
        if gt_diagnose == 1 and pred_diagnose == 0:
            missed_poscls.extend(gt_multi_label)
        if gt_diagnose == 0 and pred_diagnose == 1:
            FP_poscls.extend(pred_multi_label)
    
    total_cases = len(pred_result)

    # 统计 missed 和 FP
    missed_count = Counter(missed_poscls)
    FP_count = Counter(FP_poscls)
    total_count = Counter(total_poscls)
    stats_table = PrettyTable()
    stats_table.field_names = ["Cls ID", "Cls Name", "Total Count", "Missed Count", "FP Count"]
    sum_total, sum_missed, sum_fp = 0, 0, 0
    for cls_id, cls_name in enumerate(classes):
        tc = total_count.get(cls_id, 0)
        mc = missed_count.get(cls_id, 0)
        fc = FP_count.get(cls_id, 0)

        sum_total += tc
        sum_missed += mc
        sum_fp += fc

        stats_table.add_row([cls_id, cls_name, tc, mc, fc])
    stats_table.add_row(["-", "Total", sum_total, sum_missed, sum_fp])

    # ========== 一致性矩阵 ==========
    consistency_table = PrettyTable()
    consistency_table.field_names = ["", "multi=0", "multi>0", "Total"]

    row_pred0 = [matrix["pred_diag_0"]["pred_multilabel_0"], matrix["pred_diag_0"]["pred_multilabel_1"],
                 matrix["pred_diag_0"]["pred_multilabel_0"] + matrix["pred_diag_0"]["pred_multilabel_1"]]
    row_pred1 = [matrix["pred_diag_1"]["pred_multilabel_0"], matrix["pred_diag_1"]["pred_multilabel_1"],
                 matrix["pred_diag_1"]["pred_multilabel_0"] + matrix["pred_diag_1"]["pred_multilabel_1"]]

    consistency_table.add_row(["pred=0"] + row_pred0)
    consistency_table.add_row(["pred=1"] + row_pred1)
    consistency_table.add_row(["Total",
                               row_pred0[0] + row_pred1[0],
                               row_pred0[1] + row_pred1[1],
                               total_cases])
    # consistency_ratio: 斜对角 (pred=0,multi=0 + pred=1,multi>0) / total
    consistency_count = matrix['pred_diag_0']['pred_multilabel_0'] + matrix['pred_diag_1']['pred_multilabel_1']
    consistency_ratio = round(consistency_count / total_cases * 100, 2)

    # 打印
    print("\n=== Per-class statistics ===")
    print(stats_table)
    print("\n=== Consistency matrix ===")
    print(consistency_table)
    print(f"\nconsistency_ratio (diagonal): {consistency_ratio}%")

    # 保存到txt
    with open(save_path, "w", encoding="utf-8") as f:
        f.write("=== Per-class statistics ===\n")
        f.write(str(stats_table))
        f.write("\n\n=== Consistency matrix ===\n")
        f.write(str(consistency_table))
        f.write(f"\n\nconsistency_ratio (diagonal): {consistency_ratio}%\n")
    print(f"Error analyze result saved in {save_path}")

def visual_FN(pred_result, coco_jsonfile, visual_nums, img_savedir):
    thr = 0.5
    visual_cnt = 0
    coco = COCO(coco_jsonfile)
    classes = [i['name'] for i in coco.cats.values()]
    filename2imgid = {}
    for imgitem in tqdm(coco.imgs.values(), ncols=90, desc='Load filename2imgid'):
        filename = imgitem['file_name'].split('/')[-1]
        filename2imgid[filename] = imgitem['id']

    for item in tqdm(pred_result, ncols=90, desc='Visual FN images'):
        
        if visual_nums>0 and visual_cnt>=visual_nums:
            break
        gt_diagnose = int(len(item.gt_label)>0)
        pred_diagnose = int(item.img_prob > 0.5)
        FN_flag = gt_diagnose == 1 and pred_diagnose == 0
        if not FN_flag:
            continue
        img = Image.open(item.img_path).convert("RGB")
        filename = os.path.basename(item.img_path)
        annids = coco.getAnnIds([filename2imgid[filename]])
        annos = coco.loadAnns(annids)
        pred_multi_label = [clsidx for clsidx,cls_score in enumerate(item.pos_prob) if cls_score > thr]

        fig, ax = plt.subplots(1, figsize=(10, 10))
        ax.imshow(img)

        for ann in annos:
            bbox = ann['bbox']  # [x,y,w,h]
            cls_id = ann['category_id']
            cls_name = coco.cats[cls_id]['name']
            color = coco.cats[cls_id]['color'] if 'color' in coco.cats[cls_id] else [255, 0, 0]
            rect = patches.Rectangle(
                (bbox[0], bbox[1]), bbox[2], bbox[3],
                linewidth=2, edgecolor=[c/255 for c in color], facecolor='none'
            )
            ax.add_patch(rect)
            ax.text(
                bbox[0], bbox[1] - 2, cls_name,
                fontsize=10, color='white', backgroundcolor=[c/255 for c in color]
            )

        pred_labels_str = ", ".join([classes[idx] for idx in pred_multi_label]) if len(pred_multi_label) > 0 else "None"
        ax.set_title(f"{pred_labels_str}", fontsize=12, color='red')

        save_path = os.path.join(img_savedir, filename)
        plt.axis('off')
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()
        visual_cnt += 1


if __name__ == '__main__':
    main()

    # classes = ['AGC', 'ASC-US', 'LSIL', 'ASC-H', 'HSIL', 'SCC']
    # save_path = "log/cdetector/mlc/acc_87.92/error_analyze.txt"
    # with open("log/cdetector/mlc/acc_87.92/pred_result.pkl", "rb") as f:
    #     pred_result = pickle.load(f)
    # analyze(pred_result, classes, save_path)

    # coco_jsonfile = 'data_resource/ComparisonDetectorDataset/WINDOW_SIZE_400/annofiles/val.json'
    # visual_nums = 50    # -1 means visual all FN imgs
    # img_savedir = "log/cdetector/mlc/acc_87.92/visual_FN"
    # os.makedirs(img_savedir, exist_ok=True, mode=0o777)
    # visual_FN(pred_result, coco_jsonfile, visual_nums, img_savedir)

'''
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 --master_port=12340 scripts/analyze/mlc_error.py \
    log/cdetector/mlc/acc_87.92/config.py \
    log/cdetector/mlc/acc_87.92/checkpoints/best.pth \
    log/cdetector/mlc/acc_87.92 \
    --save_result \
    --visual_FN_flag \
    --visual_FN_cocojsonfile data_resource/ComparisonDetectorDataset/WINDOW_SIZE_400/annofiles/val.json \
    --visual_FN_nums 50
'''