import json
import torch
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torchvision.ops import nms
from cellpose import models,utils,dynamics
from mmdet.evaluation import CocoMetric
from tqdm import tqdm
from mmdet.evaluation import DumpProposals
from cerwsi.utils import flow2cellprob,inst2bboxes,gene_gridboxes,postprocess_bboxes,missed_analyze


def infer_single_img(img_RGB, cellpose_model):
    cell_config = {
        'nucleus': dict(dia=15, flowThr=0.6, cellprobThr=0.1, min_size=15),
        'cytoplasm': dict(dia=120, flowThr=0.8, cellprobThr=0.1, min_size=10*10),
        'cluster': dict(dia=240, flowThr=-1, cellprobThr=0.1, min_size=10*10),
    }

    mask_instlist = []
    for ctype,config in cell_config.items():
        flowThr,dia = config['flowThr'],float(config['dia'])
        cellprobThr,minSize = config['cellprobThr'], config['min_size']
        masks_pred, results, styles = cellpose_model.eval([img_RGB], batch_size=64, 
            flow_threshold=flowThr, diameter=dia, augment=True, compute_masks=False)
        flowi, dP, cellprob = results[0]

        if ctype == 'cytoplasm' or ctype == 'nucleus':
            cellprob,boundary_mask = flow2cellprob(dP)
            cellprob[boundary_mask] = 0.
            maski = dynamics.resize_and_compute_masks(
                    dP, cellprob,
                    cellprob_threshold=cellprobThr,
                    flow_threshold=flowThr, resize=None,
                    min_size=minSize, max_size_fraction=0.9,
                    device=cellpose_model.device)
        else:
            cellprob,boundary_mask = flow2cellprob(dP)
            cellprob[boundary_mask] = 0.
            binary = (cellprob > cellprobThr).astype(np.uint8)
            num_labels, labels = cv2.connectedComponents(binary, connectivity=8)
            # labels = postprocess(cellprob, boundary_mask.astype(float))
            maski = labels.astype(np.int32)
            maski = utils.fill_holes_and_remove_small_masks(maski, min_size=minSize)

        mask_instlist.extend(inst2bboxes(maski))
    
    total_bboxes = [item['bbox'] for item in mask_instlist]
    
    # NMS
    total_scores = [1.]*len(total_bboxes)
    bboxes = torch.tensor(total_bboxes, dtype=torch.float32)  # (N, 4)
    scores = torch.tensor(total_scores, dtype=torch.float32)  # (N,)
    nms_indices = nms(bboxes, scores, iou_threshold=0.7)
    final_bboxes = torch.tensor([total_bboxes[i] for i in nms_indices]).tolist()

    return final_bboxes

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

def infer(GPU_idx):
    cellpose_model = get_models(GPU_idx)
    for tag in ['fold1_train', 'fold1_val']:
        tag_proposal_savedir = f'{proposal_savedir}/{tag}'
        os.makedirs(tag_proposal_savedir, exist_ok=True, mode=0o777)
        jsonfile = f'{root_dir}/annofiles_roi/{tag}.json'
        with open(jsonfile, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        set_split = np.array_split(range(len(json_data['images'])), 8)
        process_group = [json_data['images'][i] for i in set_split[GPU_idx]]

        for imgitem in tqdm(process_group, ncols=80):
            purename = imgitem["file_name"].split('.')[0]
            imgpath = f'{root_dir}/JPEGImages/{purename}.png'
            img = cv2.imread(imgpath)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            final_bboxes = infer_single_img(img, cellpose_model)
            
            with open(f'{tag_proposal_savedir}/{imgitem["id"]}.json', 'w', encoding='utf-8') as f:
                json.dump(final_bboxes, f, ensure_ascii=False)

def demo_test(GPU_idx):
    cellpose_model = get_models(GPU_idx)
    purenames = [
        '1657bj008_0150','1657bj008_0066','1662bj013_0096','1657bj008_0242'
    ]
    for purename in purenames:
        imgpath = f'data_resource/HMCHH/JPEGImages/{purename}.png'
        img = cv2.imread(imgpath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        final_bboxes = infer_single_img(img, cellpose_model)
        
        visual_saveroot = 'statistic_results/cellpose_infer/hmchh_demo/0722'
        os.makedirs(visual_saveroot, exist_ok=True, mode=0o777)
        savepath = f'{visual_saveroot}/{purename}.png'
        visual_imgmask(img, final_bboxes, savepath)

def get_models(GPU_idx):
    device = torch.device(f"cuda:{GPU_idx}")
    cellpose_ckpt = 'checkpoints/cpsam'
    cellpose_model = models.CellposeModel(gpu=True, pretrained_model=cellpose_ckpt, device=device)
    return cellpose_model

def augmentBbox(bbox, maxlen=2048):
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    if w < 50 and h < 50:
        shift = 10
    elif w < 100 and h < 100:
        shift = 15
    else:
        shift = 20
    x1_new = max(0, x1 - shift)
    y1_new = max(0, y1 - shift)
    x2_new = min(maxlen - 1, x2 + shift)
    y2_new = min(maxlen - 1, y2 + shift)

    return [x1_new, y1_new, x2_new, y2_new]

def eval_metric():
    maxDet,gene_gridsize = 1000,64
    grid_boxes = gene_gridboxes(2048, 2048, gene_gridsize)
    for tag in ['fold1_train', 'fold1_val']:
        tag_proposal_savedir = f'{proposal_savedir}/{tag}'
        jsonfile = f'{root_dir}/annofiles_roi/{tag}.json'
        coco_metric = CocoMetric(
            ann_file=jsonfile,
            metric='proposal',
            classwise=False,
            iou_thrs=[0.3],
            proposal_nums=(100, 300, 1000)
        )
        coco_metric.dataset_meta = dict(classes=['abnormal'])
        with open(jsonfile, 'r', encoding='utf-8') as f:
            gt_data = json.load(f)

        for imgitem in tqdm(gt_data['images'], ncols=80):
            with open(f'{tag_proposal_savedir}/{imgitem["id"]}.json', 'r', encoding='utf-8') as f:
                proposal_bboxes = json.load(f)
            proposal_bboxes, proposal_scores = postprocess_bboxes(proposal_bboxes, grid_boxes, maxDet, minlen=10)

            pred_bboxes = torch.as_tensor([augmentBbox(bbox) for bbox in proposal_bboxes])
            pred_scores = torch.as_tensor([1.] * len(pred_bboxes))
            pred_labels = torch.as_tensor([0] * len(pred_bboxes))
            pred_instances = dict(bboxes=pred_bboxes,scores=pred_scores,labels=pred_labels)

            coco_metric.process({},[dict(pred_instances=pred_instances, 
                img_id=imgitem['id'], ori_shape=(imgitem['width'], imgitem['height']))])

        eval_results = coco_metric.evaluate(size=len(gt_data['images']))
        print(eval_results)


def foramt_proposal2coco():
    maxDet,gene_gridsize = 1000,64
    grid_boxes = gene_gridboxes(2048, 2048, gene_gridsize)
    for tag in ['fold1_train', 'fold1_val']:
        tag_proposal_savedir = f'{proposal_savedir}/{tag}'
        jsonfile = f'{root_dir}/annofiles_roi/{tag}.json'
        with open(jsonfile, 'r', encoding='utf-8') as f:
            json_data = json.load(f)

        predinfo = []
        for imgitem in tqdm(json_data['images'], ncols=80):
            with open(f'{tag_proposal_savedir}/{imgitem["id"]}.json', 'r', encoding='utf-8') as f:
                proposal_bboxes = json.load(f)
            proposal_bboxes, proposal_scores = postprocess_bboxes(proposal_bboxes, grid_boxes, maxDet, minlen=10)
            for bbox in proposal_bboxes:
                x1,y1,x2,y2 = augmentBbox(bbox)
                w, h = x2-x1, y2-y1
                predinfo.append({
                    'image_id': imgitem['id'],
                    'category_id': 1,
                    'score': 1.,
                    'bbox': torch.tensor([x1, y1, w, h]).tolist()
                })
        with open(f'{proposal_savedir}/{tag}_maxDet{maxDet}.json', 'w', encoding='utf-8') as f:
            json.dump(predinfo, f, ensure_ascii=False)

def foramt_proposal2pkl():
    maxDet,gene_gridsize = 1000,64
    grid_boxes = gene_gridboxes(2048, 2048, gene_gridsize)
    for tag in ['fold1_train', 'fold1_val']:
        tag_proposal_savedir = f'{proposal_savedir}/{tag}'
        jsonfile = f'{root_dir}/annofiles_roi/{tag}.json'
        with open(jsonfile, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        dump_handle = DumpProposals(
            output_dir = f'{proposal_savedir}/',
            proposals_file = f'{tag}_maxdet{maxDet}.pkl',
            num_max_proposals = maxDet
        )
        for imgitem in tqdm(json_data['images'], ncols=80):
            with open(f'{tag_proposal_savedir}/{imgitem["id"]}.json', 'r', encoding='utf-8') as f:
                proposal_bboxes = json.load(f)
            proposal_bboxes, proposal_scores = postprocess_bboxes(proposal_bboxes, grid_boxes, maxDet, minlen=10)
            shift_bboxes = torch.tensor([augmentBbox(bbox) for bbox in proposal_bboxes])
            dump_handle.process(None, [{
                'pred_instances': dict(bboxes=shift_bboxes,scores=proposal_scores),
                'img_path': f'data_resource/HMCHH/JPEGImages/{imgitem["file_name"]}'
            }])

        dump_handle.evaluate(size=len(json_data['images']))

if __name__ == "__main__":
    root_dir = 'data_resource/HMCHH'
    proposal_savedir = f'{root_dir}/proposals_file/cp_0722'
    os.makedirs(proposal_savedir, exist_ok=True, mode=0o777)
    GPU_idx = 7
    demo_test(GPU_idx)
    # infer(GPU_idx)
    # eval_metric()

    # foramt_proposal2coco()
    # visual_saveroot = 'statistic_results/cellpose_infer/hmchh_missed_0722'
    # os.makedirs(visual_saveroot, exist_ok=True, mode=0o777)
    # missed_analyze(
    #     f'{root_dir}/annofiles_roi/fold1_train.json',
    #     f'{proposal_savedir}/fold1_train_maxDet1000.json',
    #     visual_saveroot = visual_saveroot,
    #     image_root = 'data_resource/HMCHH/JPEGImages'
    # )

    # foramt_proposal2pkl()

'''
 Average Recall     (AR) @[ IoU=0.30:0.30 | area=   all | maxDets=100 ] = 0.289
 Average Recall     (AR) @[ IoU=0.30:0.30 | area=   all | maxDets=300 ] = 0.728
 Average Recall     (AR) @[ IoU=0.30:0.30 | area=   all | maxDets=1000 ] = 0.961
 Average Recall     (AR) @[ IoU=0.30:0.30 | area= small | maxDets=1000 ] = 0.286
 Average Recall     (AR) @[ IoU=0.30:0.30 | area=medium | maxDets=1000 ] = 0.968
 Average Recall     (AR) @[ IoU=0.30:0.30 | area= large | maxDets=1000 ] = 0.960
'''