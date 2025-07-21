import json
import torch
from cerwsi.utils import build_evaluator,ImgODCOCOMetric
from mmengine.config import Config
from tqdm import tqdm
from cerwsi.datasets import load_data

def main():
    dataset_config_file = 'configs/dataset/mmdet/l_cerscanv1_dataset.py'
    d_cfg = Config.fromfile(dataset_config_file)
    json_save_dir = f'{d_cfg.data_root}/sam2Infer'
    logger_name = 'eval_info'
    evaluator = build_evaluator([ImgODCOCOMetric(logger_name,'',d_cfg.val_evaluator,d_cfg.classes)])
    # evaluator = build_evaluator([ImgODMetric(logger_name,'')])
    
    dataloader = load_data(d_cfg, ['train'])
    error_filenames = []
    for i_batch, sampled_batch in enumerate(tqdm(dataloader, ncols=70)):
        pred_bboxes,img_probs = [],[]
        for datasample in sampled_batch['data_samples']:
            filename = datasample.img_path.split('/')[-1]
            purename = filename.replace('.png', '')
            prefix = datasample.extra_info['prefix']
            save_jsonname = f'{json_save_dir}/{prefix}/{purename}.json'
            try:
                with open(save_jsonname, 'r', encoding='utf-8') as f:
                    masks = json.load(f)
            except:
                print(f'JSON load error: {purename}')
                error_filenames.append(purename)
    print(len(error_filenames))
    #         image_boxes = []
    #         for maskinfo in masks:
    #             x1, y1, w, h = maskinfo['bbox']
    #             image_boxes.append({
    #                 'bbox': [x1, y1, x1+w, y1+h],
    #                 'score': maskinfo['stability_score'],
    #                 'cls': 1
    #             })
            
    #         pred_bboxes.append(image_boxes)
    #         img_probs.append(1.)
    #     sampled_batch['pred_bbox'] = pred_bboxes
    #     sampled_batch['img_probs'] = torch.Tensor(img_probs)
    #     evaluator.process(data_samples=[sampled_batch], data_batch=None)
    # metrics = evaluator.evaluate(len(dataloader.dataset))
    # print(metrics)

if __name__ == "__main__":
    main()

'''
CocoMetric
 Average Precision  (AP) @[ IoU=0.50:0.50 | area=   all | maxDets=100 ] = 0.004
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.004
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1000 ] = -1.000
 Average Precision  (AP) @[ IoU=0.50:0.50 | area= small | maxDets=1000 ] = 0.001
 Average Precision  (AP) @[ IoU=0.50:0.50 | area=medium | maxDets=1000 ] = 0.007
 Average Precision  (AP) @[ IoU=0.50:0.50 | area= large | maxDets=1000 ] = 0.005
 Average Recall     (AR) @[ IoU=0.50:0.50 | area=   all | maxDets=100 ] = 0.567
 Average Recall     (AR) @[ IoU=0.50:0.50 | area=   all | maxDets=300 ] = 0.668
 Average Recall     (AR) @[ IoU=0.50:0.50 | area=   all | maxDets=1000 ] = 0.669
 Average Recall     (AR) @[ IoU=0.50:0.50 | area= small | maxDets=1000 ] = 0.673
 Average Recall     (AR) @[ IoU=0.50:0.50 | area=medium | maxDets=1000 ] = 0.631
 Average Recall     (AR) @[ IoU=0.50:0.50 | area= large | maxDets=1000 ] = 0.771

+---------------------+-----------------+---------------------+-----------------+---------------------+-----------------+
|    Thr = 0.3 key    | Thr = 0.3 value |    Thr = 0.5 key    | Thr = 0.5 value |    Thr = 0.7 key    | Thr = 0.7 value |
+---------------------+-----------------+---------------------+-----------------+---------------------+-----------------+
|     consistency     |       1.0       |     consistency     |       1.0       |     consistency     |       1.0       |
|   pp_binary_recall  |      0.9231     |   pp_binary_recall  |      0.9231     |   pp_binary_recall  |      0.9231     |
|    pp_cls_recall    |       0.0       |    pp_cls_recall    |       0.0       |    pp_cls_recall    |       0.0       |
| tp_binary_precision |      0.0173     | tp_binary_precision |      0.0173     | tp_binary_precision |      0.0173     |
|   tp_binary_recall  |      0.7838     |   tp_binary_recall  |      0.7838     |   tp_binary_recall  |      0.7838     |
|   tp_cls_precision  |      0.0001     |   tp_cls_precision  |      0.0001     |   tp_cls_precision  |      0.0001     |
|    tp_cls_recall    |      0.0057     |    tp_cls_recall    |      0.0057     |    tp_cls_recall    |      0.0057     |
+---------------------+-----------------+---------------------+-----------------+---------------------+-----------------+

 Average Precision  (AP) @[ IoU=0.30:0.30 | area=   all | maxDets=100 ] = 0.005
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = -1.000
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1000 ] = -1.000
 Average Precision  (AP) @[ IoU=0.30:0.30 | area= small | maxDets=1000 ] = 0.001
 Average Precision  (AP) @[ IoU=0.30:0.30 | area=medium | maxDets=1000 ] = 0.010
 Average Precision  (AP) @[ IoU=0.30:0.30 | area= large | maxDets=1000 ] = 0.006
 Average Recall     (AR) @[ IoU=0.30:0.30 | area=   all | maxDets=100 ] = 0.672
 Average Recall     (AR) @[ IoU=0.30:0.30 | area=   all | maxDets=300 ] = 0.783
 Average Recall     (AR) @[ IoU=0.30:0.30 | area=   all | maxDets=1000 ] = 0.783
 Average Recall     (AR) @[ IoU=0.30:0.30 | area= small | maxDets=1000 ] = 0.720
 Average Recall     (AR) @[ IoU=0.30:0.30 | area=medium | maxDets=1000 ] = 0.764
 Average Recall     (AR) @[ IoU=0.30:0.30 | area= large | maxDets=1000 ] = 0.859
'''