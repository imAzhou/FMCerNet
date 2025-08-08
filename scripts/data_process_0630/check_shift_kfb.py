import os
import numpy as np
import pandas as pd
import json
from tqdm import tqdm
from PIL import Image, ImageDraw
from datetime import datetime
from cerwsi.utils import KFBSlide,is_bbox_inside,random_cut_square
from scripts.data_process_0511.utils import imgName2patientId

SAFE_MARGIN = 100

def visual_polygon(slide, polygon_annlist, patientId, max_xy, timetag):
    [child.update({'inst_draw': False}) for child in polygon_annlist]

    imgid = 0
    for annitem in polygon_annlist:
        if annitem['inst_draw']:
            continue
        
        all_x,all_y = [p[0] for p in annitem['points']], [p[1] for p in annitem['points']]
        x1,x2 = min(all_x),max(all_x)
        y1,y2 = min(all_y),max(all_y)
        if max_xy is not None:
            x2,y2 = min(max_xy[0], x2), min(max_xy[1], y2)

        sq_size = 1024
        square_x1,square_y1 = random_cut_square((x1,y1,x2-x1,y2-y1), sq_size)
        square_x1,square_y1 = max(0,square_x1),max(0,square_y1)
        square_x2,square_y2 = square_x1+sq_size,square_y1+sq_size
        square_x2,square_y2 = min(max_xy[0], square_x2), min(max_xy[1], square_y2)

        square_w,square_h = square_x2-square_x1, square_y2-square_y1
        location, level, size = (square_x1,square_y1), 0, (square_w,square_h)
        cropped = Image.fromarray(slide.read_region(location, level, size))
        draw = ImageDraw.Draw(cropped)
        imgid += 1

        parent_patch_coords = [square_x1,square_y1,square_x2,square_y2]
        for _annitem in polygon_annlist:
            all_x,all_y = [p[0] for p in _annitem['points']], [p[1] for p in _annitem['points']]
            x1,x2 = min(all_x),max(all_x)
            y1,y2 = min(all_y),max(all_y)
            if max_xy is not None:
                x2,y2 = min(max_xy[0], x2), min(max_xy[1], y2)
            if (not _annitem['inst_draw']) and is_bbox_inside([x1,y1,x2,y2],parent_patch_coords,tolerance=10):
                polygon = list(zip(
                    (np.array(all_x) - square_x1).tolist(), 
                    (np.array(all_y) - square_y1).tolist()))
                draw.polygon(polygon, outline="red", width=3)   # 只画边框
                _annitem['inst_draw'] = True

        savedir = f'statistic_results/0630/check_shift_rectify/{timetag}/{patientId}'
        os.makedirs(savedir, exist_ok=True, mode=0o777)
        cropped.save(f"{savedir}/{imgid}.png")


def gene_zheyislide_filter():
    '''
    情况1：阳性标注框都在 RoI 内
    情况2：整个WSI没有绘制 RoI，则未标注阳性框的区域都是阴性
    情况3：WSI 有 RoI，RoI外也有部分阳性框
    '''
    df_data_0409 = pd.read_csv('data_resource/zheyi_annofiles/0409_slide_anno.csv')
    df_data_0422 = pd.read_csv('data_resource/zheyi_annofiles/0422_slide_anno.csv')
    df_data_0607 = pd.read_csv('data_resource/zheyi_annofiles/0607_slide_anno.csv')
    df_abandon = pd.read_csv('data_resource/group_csv/abandon.csv')
    df_data = pd.concat([df_data_0409, df_data_0422, df_data_0607])

    name2PID = imgName2patientId(df_data)
    timetags = ['0422','0607']

    shape_list = []
    for timetag in timetags:
        with open(f'data_resource/zheyi_annofiles/宫颈液基细胞—Slide-{timetag}.json', 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        if timetag == '0607':
            del json_data['NILM'] # 暂时不处理阴性 slide 的标注信息
        with open(f'data_resource/zheyi_annofiles/{timetag}_shift.json', 'r', encoding='utf-8') as f:
            shift_config = json.load(f)

        slideinfos = []
        for i in json_data.values():
            slideinfos.extend(i)
        for item in tqdm(slideinfos, ncols=90, desc=f'Process timetag: {timetag}'):
            patientId = name2PID[item['imageName']]
            if patientId != '' and patientId not in df_abandon['patientId']:
                rowInfo = df_data[df_data['patientId'] == patientId].iloc[0]
                slide = KFBSlide(rowInfo['kfb_path'])
                swidth, sheight = slide.level_dimensions[0]
                max_xy = (swidth-SAFE_MARGIN, sheight-SAFE_MARGIN)
                item['annotations'] = sorted(
                    item['annotations'],
                    key=lambda d: datetime.strptime(d['updatedTime'], "%Y-%m-%dT%H:%M:%SZ"),
                    reverse=True
                )
                annlist = item['annotations'][0]['annotationResult']
                polygon_annlist = []
                for annitem in annlist:
                    shape = annitem['shape']
                    if shape == 'polygonPath':
                        shift = shift_config[patientId]
                        annitem['points'] = [[p[0]-shift, p[1]-shift]for p in annitem['points']]
                        polygon_annlist.append(annitem)
                    shape_list.append(shape)
                if len(polygon_annlist) > 0:
                    visual_polygon(slide, polygon_annlist, patientId, max_xy, timetag)
    print(list(set(shape_list)))


def make_shift_file():
    df_data_0409 = pd.read_csv('data_resource/zheyi_annofiles/0409_slide_anno.csv')
    df_data_0422 = pd.read_csv('data_resource/zheyi_annofiles/0422_slide_anno.csv')
    df_data_0607 = pd.read_csv('data_resource/zheyi_annofiles/0607_slide_anno.csv')
    df_abandon = pd.read_csv('data_resource/group_csv/abandon.csv')
    df_data = pd.concat([df_data_0409, df_data_0422, df_data_0607])

    name2PID = imgName2patientId(df_data)
    timetag_config = {
        '0409': {'polygon_shift': 50, 'specify': ''},
        '0422': {'polygon_shift': 25, 'specify': 'data_resource/zheyi_annofiles/0422_shift.txt'},
        '0607': {'polygon_shift': 50, 'specify': 'data_resource/zheyi_annofiles/0607_shift.txt'},
    }
    for timetag, configs in timetag_config.items():
        with open(f'data_resource/zheyi_annofiles/宫颈液基细胞—Slide-{timetag}.json', 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        if timetag == '0607':
            del json_data['NILM'] # 暂时不处理阴性 slide 的标注信息
        
        slideinfos = []
        for i in json_data.values():
            slideinfos.extend(i)
        exist_lines = {}
        if configs['specify'] != '':
            with open(configs['specify'], 'r', encoding='utf-8') as f:
                for line in f.readlines():
                    pid = line.split(' ')[0]
                    shift_num = int(line.split(' ')[1].strip())
                    exist_lines[pid] = shift_num

        shift_txtlines = {}
        for item in tqdm(slideinfos, ncols=90, desc=f'Process timetag: {timetag}'):
            patientId = name2PID[item['imageName']]
            if patientId != '' and patientId not in df_abandon['patientId']:
                shift_txtlines[patientId] = exist_lines.get(patientId, configs['polygon_shift'])
        
        with open(f'data_resource/zheyi_annofiles/{timetag}_shift.json', 'w', encoding='utf-8') as f:
            json.dump(shift_txtlines, f, ensure_ascii=False)

if __name__ == "__main__":
    # make_shift_file()
    gene_zheyislide_filter()