import json
import warnings
import os

from tqdm import tqdm
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'
warnings.filterwarnings("ignore", category=UserWarning)
import torch
import cv2
import matplotlib.pyplot as plt
from cerwsi.nets import CellposeNet
import argparse
from torchvision.ops import nms
from cerwsi.utils import set_seed, init_distributed_mode, is_main_process
import torch.distributed as dist
from mmdet.evaluation import CocoMetric
from mmdet.evaluation import DumpProposals

# ====================== cellpose infer params ======================
SEED = 1234
tile_test_bs = 128
cellpose_ckpt = 'checkpoints/cpsam'
cervical_cell_config = {
    'nucleus': dict(dia=15, flowThr=0.6, cellprobThr=0.1, min_size=15),
    'cytoplasm': dict(dia=120, flowThr=0.8, cellprobThr=0.1, min_size=10*10),
    'cluster': dict(dia=240, flowThr=-1, cellprobThr=0.1, min_size=10*10),
}
blood_cell_config = {
    'nucleus': dict(dia=15, flowThr=0.6, cellprobThr=0.1, min_size=15),
    'cytoplasm': dict(dia=100, flowThr=0.8, cellprobThr=0.1, min_size=10*10),
    'cluster': dict(dia=150, flowThr=-1, cellprobThr=0.1, min_size=10*10),
}

dataset_config = {
    'CDetector': {
        'dataroot_dir': 'data_resource/ComparisonDetectorDataset',
        'infer_imgdir': 'test',
        'metric_json': 'test_filter_error.json',
        'cell_config': cervical_cell_config
    },
    'HMCHH': {
        'dataroot_dir': 'data_resource/HMCHH',
        'infer_imgdir': 'JPEGImages',
        'metric_json': 'annofiles/fold1_val.json',
        'cell_config': cervical_cell_config
    },
    'CRIC': {
        'dataroot_dir': 'data_resource/CRIC',
        'infer_imgdir': 'images',
        'metric_json': 'annofiles/abnormal/fold4_train.json',
        'cell_config': cervical_cell_config
    },
    'BCCD':{
        'dataroot_dir': 'data_resource/BCCD',
        'infer_imgdir': 'train',
        'metric_json': 'annofiles/train_annotations.coco.json',
        'cell_config': blood_cell_config
    },
    'WS1600': {
        'dataroot_dir': 'data_resource/WINDOW_SIZE_1600',
        'infer_imgdir': 'images/total_pos',
        # 'metric_json': 'annofiles/puretrain_cocoformat.json'
        'metric_json': 'annofiles/total_cocoformat.json',
        'cell_config': cervical_cell_config
    },
}

DATASET_TAG = 'CRIC'

dataroot_dir = dataset_config[DATASET_TAG]['dataroot_dir']
infer_imgdir = dataset_config[DATASET_TAG]['infer_imgdir']
metric_json = dataset_config[DATASET_TAG]['metric_json']
cell_config = dataset_config[DATASET_TAG]['cell_config']

infer_imgdirs = [
    f'{dataroot_dir}/{infer_imgdir}',
]
infer_savedir = f'{dataroot_dir}/cellpose_all'
os.makedirs(infer_savedir, exist_ok=True, mode=0o777)

# =================== infer results metric params ===================
iou_thrs = [0.3]
metric_jsons = [
    f'{dataroot_dir}/{metric_json}',
]
# ================== infer results format to pkl params ==================
KEEPNUM = 300
proposal_pkl_cfg = {
    'source_cocojson': metric_jsons[0], #拼接成得路径应为data_resource/WINDOW_SIZE_1600/annofiles/total_cocoformat.json
    'output_dir': dataroot_dir,
    'pkl_filename': f'fold4_train_proposal_maxDet{KEEPNUM}.pkl',
    'img_dir': f'{dataroot_dir}/{infer_imgdir}',
}
# ================== infer results demo visual params ==================
demo_nums = -1  # < 0 时不绘制结果, > 0 时绘制完 demo_nums 程序会自动结束
demoimg_savedir = f'statistic_results/cellpose_infer/{DATASET_TAG}'
os.makedirs(demoimg_savedir, exist_ok=True, mode=0o777)


def visual_imgmask(img_RGB, bboxeslist, savepath):
    plt.figure(figsize=(10, 10))
    ax = plt.gca()
    plt.imshow(img_RGB)
    for (x1,y1,x2,y2) in bboxeslist:
        rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, edgecolor='lime', linewidth=2, facecolor='none')
        ax.add_patch(rect)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(savepath)
    plt.close()


def run_inference(cellpose_model):
    total_img_paths = []
    for dirname in infer_imgdirs:
        img_paths = [f'{dirname}/{filename}' for filename in os.listdir(dirname)]
        total_img_paths.extend(img_paths)
    if demo_nums > 0:
        total_img_paths = total_img_paths[:demo_nums]

    if is_main_process():
        done_nums = len(os.listdir(infer_savedir))
        print(f"\n{'='*40}")
        print(f'Total images: {len(total_img_paths)}')
        print(f'Resumed {done_nums}, Left {len(total_img_paths)-done_nums}.')
        print(f"{'='*40}")
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    data_per_rank = total_img_paths[rank::world_size]

    for p_idx,imgpath in enumerate(data_per_rank):
        filename = os.path.basename(imgpath)
        purename = os.path.splitext(os.path.basename(filename))[0]
        if os.path.exists(f'{infer_savedir}/{purename}.json'):
            continue
        
        img = cv2.imread(imgpath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        object_list = cellpose_model(img, batchsize=tile_test_bs)
        
        total_bboxes = [item['bbox'] for item in object_list]
        # NMS
        bboxes = torch.tensor(total_bboxes, dtype=torch.float32)  # (N, 4)
        widths = bboxes[:, 2] - bboxes[:, 0]
        heights = bboxes[:, 3] - bboxes[:, 1]
        scores = widths * heights  # 面积越大，得分越高
        nms_indices = nms(bboxes, scores, iou_threshold=0.5)
        final_bboxes = bboxes[nms_indices].tolist()

        if demo_nums > 0:
            savepath = f'{demoimg_savedir}/{purename}.png'
            visual_imgmask(img, final_bboxes, savepath)

        with open(f'{infer_savedir}/{purename}.json', 'w', encoding='utf-8') as f:
            json.dump(final_bboxes, f, ensure_ascii=False)

        print(f'\r[Rank {rank}] Processed {p_idx+1}/{len(data_per_rank)} ', end='')


def get_models(device, gpu):
    cellpose_model = CellposeNet(cell_config, userle=True).to(device)
    cellpose_model.load_ckpt(cellpose_ckpt)
    cellpose_model = torch.nn.parallel.DistributedDataParallel(
        cellpose_model, device_ids=[gpu], find_unused_parameters=False)
    cellpose_model = cellpose_model.module
    
    return cellpose_model

def main():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    init_distributed_mode(args)
    set_seed(SEED)
    device = torch.device(f'cuda:{os.getenv("LOCAL_RANK")}')
    cellpose_model = get_models(device,args.gpu)
    
    run_inference(cellpose_model)
    torch.distributed.destroy_process_group()
       

def eval_metric():
    for jsonfile in metric_jsons:
        with open(jsonfile, 'r', encoding='utf-8') as f:
            gt_data = json.load(f)
        for thr in iou_thrs:
            coco_metric = CocoMetric(
                ann_file=jsonfile,
                metric='proposal',
                classwise=False,
                iou_thrs=[thr],
                proposal_nums=(100, 300, 1000)
            )
            classes = [i['name'] for i in gt_data['categories']]
            coco_metric.dataset_meta = dict(classes=classes)

            for imgitem in tqdm(gt_data['images'], ncols=80, desc='Eval Metric'):
                filename = imgitem['file_name']
                purename = os.path.splitext(os.path.basename(filename))[0]
                # if purename != '00112':
                #     continue
                with open(f'{infer_savedir}/{purename}.json', 'r', encoding='utf-8') as f:
                    proposal_bboxes = json.load(f)

                # 按面积从大到小排序
                final_bboxes_sorted = sorted(
                    proposal_bboxes,
                    key=lambda box: (box[2] - box[0]) * (box[3] - box[1]),
                    reverse=True
                )
                final_bboxes = final_bboxes_sorted[:KEEPNUM]

                pred_bboxes = torch.as_tensor(final_bboxes)
                pred_scores = torch.as_tensor([1.] * len(pred_bboxes))
                pred_labels = torch.as_tensor([0] * len(pred_bboxes))
                pred_instances = dict(bboxes=pred_bboxes,scores=pred_scores,labels=pred_labels)

                coco_metric.process({},[dict(pred_instances=pred_instances, 
                    img_id=imgitem['id'], ori_shape=(imgitem['width'], imgitem['height']))])

            eval_results = coco_metric.evaluate(size=len(gt_data['images']))
            print(f'Eval thr={thr} in {jsonfile}:')
            print(eval_results)


def foramt_proposal2pkl():
    jsonfile = proposal_pkl_cfg['source_cocojson']
    output_dir = proposal_pkl_cfg['output_dir']
    pkl_filename = proposal_pkl_cfg['pkl_filename'] #转换好的pkl文件得路径
    img_dir = proposal_pkl_cfg['img_dir']
    with open(jsonfile, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    dump_handle = DumpProposals(
        output_dir = output_dir,
        proposals_file = pkl_filename,
        num_max_proposals = KEEPNUM
    )
    for imgitem in tqdm(json_data['images'], ncols=80):
        filename = imgitem['file_name']
        # purename = filename.split('.')[0]

        # 使用 os.path 模块提取纯文件名，确保去除所有路径信息和后缀
        # 示例: 'total_pos/JFSW_..._1.png' -> 'JFSW_..._1'
        purename = os.path.splitext(os.path.basename(filename))[0]

        with open(f'{infer_savedir}/{purename}.json', 'r', encoding='utf-8') as f:
            proposal_bboxes = json.load(f)  # (x1,y1,x2,y2)
        
        # 按面积从大到小排序
        final_bboxes_sorted = sorted(
            proposal_bboxes,
            key=lambda box: (box[2] - box[0]) * (box[3] - box[1]),
            reverse=True
        )
        final_bboxes = final_bboxes_sorted[:KEEPNUM]
        if len(final_bboxes) < KEEPNUM:
            extra_nums = KEEPNUM - len(final_bboxes)
            # random_boxes = generate_random_bboxes(imgh, imgw, extra_nums, min_size=50)
            imgw,imgh = imgitem['width'],imgitem['height']
            final_bboxes.extend([[0,0,imgw,imgh] for i in range(extra_nums)])

        proposal_bboxes = torch.as_tensor(final_bboxes)
        proposal_scores = torch.as_tensor([1.] * len(final_bboxes))
        dump_handle.process(None, [{
            'pred_instances': dict(bboxes=proposal_bboxes,scores=proposal_scores),
            'img_path': f'{img_dir}/{imgitem["file_name"]}'
        }])

    dump_handle.evaluate(size=len(json_data['images']))

if __name__ == '__main__':
    # main()
    eval_metric()
    foramt_proposal2pkl()

'''
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 --master_port=12341 scripts/everything_mode/cellpose_1017.py
'''

