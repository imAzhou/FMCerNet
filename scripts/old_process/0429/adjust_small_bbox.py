import glob
import os
import json
from tqdm import tqdm
from PIL import Image
from cerwsi.utils import draw_OD,is_bbox_inside,calc_relative_coord,generate_cut_regions,random_cut_square


POSITIVE_CLASS = ['AGC', 'ASC-US','LSIL', 'ASC-H', 'HSIL']
invalid_annos = [os.path.basename(i) for i in glob.glob('statistic_results/0429/adjust_small_bbox/**/*.png')]


def load_annofile():
    ann_json = 'data_resource/0328/annofiles/宫颈液基细胞—RoI_filter.json'
    with open(ann_json,'r') as f:
        annotaion = json.load(f)
    small_cnt = 0
    for idx,imgitem in enumerate(tqdm(annotaion, ncols=80)):
        if idx < 87:
            continue
        filename = imgitem['imageName']
        patientId = filename.replace('_RoI.png','')
        img_path = f'data_resource/0328/{imgitem["tag"]}/img/{filename}'
        img = Image.open(img_path)
        # filter_annitem = []
        for aidx,annitem in enumerate(imgitem['annotations']):
            anno_idxname = f'{patientId}_{aidx}.png'
            if anno_idxname in invalid_annos:
                continue
            sub_class = annitem['label']
            if annitem['type'] != 'circle' and sub_class in POSITIVE_CLASS:
                all_x,all_y = [p[0] for p in annitem['points']],[p[1] for p in annitem['points']]
                x1,x2 = min(all_x),max(all_x)
                y1,y2 = min(all_y),max(all_y)

                # filter_annitem.append(annitem)
                w,h = x2-x1, y2-y1
                if w < 50 or h < 50:
                    save_dir = f'statistic_results/0429/adjust_small_bbox/{patientId}'
                    os.makedirs(save_dir, exist_ok=True)
                    small_cnt += 1
                    start_x, start_y = random_cut_square((x1,y1,w,h), 400)
                    sx1,sy1,sx2,sy2 = start_x, start_y, start_x+400, start_y+400
                    cropped = img.crop((sx1,sy1,sx2,sy2))
                    bbox_coord = calc_relative_coord([sx1,sy1,sx2,sy2], [x1, y1, x2, y2])
                    save_path = f'{save_dir}/{patientId}_{aidx}.png'
                    draw_OD(cropped,save_path,[0,0,400,400], 
                            [dict(sub_class=annitem['label'],region=bbox_coord)], POSITIVE_CLASS)
    #         imgitem['annotations'] = filter_annitem
    #         imgitem['tag'] = tag
    #         filter_anno_list.append(imgitem)
    
    # filter_ann_json = 'data_resource/0328/annofiles/宫颈液基细胞—RoI_filter.json'
    # with open(filter_ann_json, 'w', encoding='utf-8') as f:
    #     json.dump(filter_anno_list, f, ensure_ascii=False, indent=4)

    # print(small_cnt)

def main():
    total_annos = load_annofile()

if __name__ == "__main__":
    main()