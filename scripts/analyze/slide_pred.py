import pickle
import os
from tqdm import tqdm
from mmengine.config import Config
from collections import Counter, defaultdict
from prettytable import PrettyTable


def analyze(pred_result, classes, save_dir):
    num_classes = len(classes)
    # 1. 被误分类为阴性的阳性 case
    false_negative_counter = Counter()
    # 2. 每个GT类别的错误预测分布
    misclass_counter = defaultdict(Counter)
    # 统计每个GT类别的总样本数
    gt_counter = Counter()
    each_slide_lines = []
    for item in tqdm(pred_result, ncols=80):
        cls_gt_label = item.slide_label       # GT类别 (int)
        gt_counter[cls_gt_label] += 1
        pn_gt_label = int(cls_gt_label > 0)   # 阴阳 GT标签
        
        cls_pred_label = item.pred_label.item()   # 预测类别 (int)
        pn_pred_label = int(cls_pred_label > 0)   # 阴阳预测标签

        # (1) 阳性被误判为阴性
        if pn_gt_label == 1 and pn_pred_label == 0:
            false_negative_counter[cls_gt_label] += 1
        # (2) 每个类别的误分类统计
        if cls_gt_label != cls_pred_label:
            misclass_counter[cls_gt_label][cls_pred_label] += 1
        pid = item.slide_info['patientId']
        kfb_clsname = item.slide_info['kfb_clsname']
        pred_clsname = item.pred_clsname
        filename = os.path.basename(item.slide_info['kfb_path'])
        each_slide_lines.append(f"{pid} filename({filename}) GT({kfb_clsname}) Pred({pred_clsname})\n")
    
    slide_pred_savepath = f"{save_dir}/each_slide_pred.txt"
    with open(slide_pred_savepath, "w", encoding="utf-8") as f:
        f.writelines(each_slide_lines)
    print(f"Each slide pred result saved in {slide_pred_savepath}")

    analyze_savepath = f"{save_dir}/error_analyze.txt"
    with open(analyze_savepath, "w", encoding="utf-8") as f:
        # 1. 被误分类为阴性的阳性 case 各类别分布
        table_fn = PrettyTable()
        table_fn.field_names = ["GT类别", "数量"]
        for cls_id in range(1, num_classes):   # 0 是阴性，跳过
            table_fn.add_row([classes[cls_id], false_negative_counter[cls_id]])
        f.write("【误判为阴性的阳性 case 各类别分布】\n")
        # f.write(table_fn.get_string() + "\n\n")
        f.write(str(table_fn) + "\n\n")

        # 2. 每个GT类别的误分类情况
        f.write("【逐类别误分类分布】\n")
        for cls_id in range(1, num_classes):   # 遍历阳性类别
            table_mis = PrettyTable()
            table_mis.field_names = ["GT类别", "预测为", "数量"]
            total_mis = sum(misclass_counter[cls_id].values())
            total_samples = gt_counter[cls_id]  # 该类别总样本数
            if total_mis == 0:
                continue
            sorted_items = sorted(
                misclass_counter[cls_id].items(), key=lambda x: x[1], reverse=True
            )
            for pred_id, cnt in sorted_items:
                table_mis.add_row([classes[cls_id], classes[pred_id], cnt])
            f.write(
                f"\nGT类别 = {classes[cls_id]} "
                f"(总样本数 {total_samples}, 总误分类 {total_mis})\n"
            )
            f.write(table_mis.get_string() + "\n")

    print(f"Analyze result saved in {analyze_savepath}")


if __name__ == '__main__':
    log_root_dir = 'log/slide_mc/ours_WS1600/2025_09_10_15_50_31'
    gt_csvfiles = [
        'data_resource/0630/WINDOW_SIZE_1600/annofiles/45_0907_train.csv',
        'data_resource/0630/WINDOW_SIZE_1600/annofiles/67_0907_val.csv'
    ]
    pred_results = [
        f"{log_root_dir}/pred_result_train.pkl",
        f"{log_root_dir}/pred_result_val.pkl",
    ]

    cfg = Config.fromfile(f'{log_root_dir}/config.py')
    total_pred_result = []
    for pkl_file in pred_results:
        with open(pkl_file, "rb") as f:
            pred_result = pickle.load(f)
            total_pred_result.extend(pred_result)
    analyze(total_pred_result, cfg.classes, log_root_dir)


