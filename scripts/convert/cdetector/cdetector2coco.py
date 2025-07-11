import json
import cv2
import os
from pycocotools.coco import COCO
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from tqdm import tqdm
import random
from cerwsi.utils import generate_cut_regions,is_bbox_inside

POSITIVE_CLASS = ['ASC-US', 'LSIL', 'ASC-H', 'HSIL', 'AGC']
classes = ['negative', *POSITIVE_CLASS]
WINDOW_SIZE = 406   # 要保证 WINDOW_SIZE 是 featmap_size 的整数倍
STRIDE = 350
featmap_size = 14


def get_cutregion_inside(square_coord, inside_items, min_overlap=50):
    '''
    Args:
        square_coord: patch块在原图中的坐标(x1, y1, x2, y2)
        inside_items: 原图中所有的标注框信息
        min_overlap: 标注框与patch块的最小交集阈值，若大于该阈值，则此标注框属于该patch块
    Return:
        item_inside_filter: 在patch块中或与patch块有交集区域的标注框信息，且标注框坐标已经更新为相对于patch块左上角的相对坐标, eg: [dict(sub_class='ASC-US', region=dict(x=0, y=0, width=66, height=77))]
    '''
    item_inside_filter = []
    for idx,i in enumerate(inside_items):
        region = i.get('region')
        w,h = region['width'],region['height']
        x1,y1 = region['x'],region['y']
        x2,y2 = x1+w, y1+h
        if w<5 or h<5:
            continue
        sx1,sy1,sx2,sy2 = square_coord
        
        if is_bbox_inside([x1,y1,x2,y2], square_coord, tolerance=5):
            relative_bbox = [x1 - sx1, y1 - sy1, x2 - sx1, y2 - sy1]
            rx1,ry1,rx2,ry2 = relative_bbox
            relative_region = dict(x=rx1, y=ry1, width=rx2-rx1, height=ry2-ry1)
            item_inside_filter.append(dict(sub_class=i['sub_class'], region=relative_region))
            continue
        
        # 计算交集区域
        inter_x_min = max(x1, sx1)
        inter_y_min = max(y1, sy1)
        inter_x_max = min(x2, sx2)
        inter_y_max = min(y2, sy2)

        if inter_x_min < inter_x_max and inter_y_min < inter_y_max:
            inter_w = inter_x_max - inter_x_min
            inter_h = inter_y_max - inter_y_min
            if inter_w > min_overlap and inter_h > min_overlap:
                relative_bbox = [inter_x_min - sx1, inter_y_min - sy1, inter_x_max - sx1, inter_y_max - sy1]
                rx1,ry1,rx2,ry2 = relative_bbox
                relative_region = dict(x=rx1, y=ry1, width=rx2-rx1, height=ry2-ry1)
                item_inside_filter.append(dict(sub_class=i['sub_class'], region=relative_region))
    
    return item_inside_filter

def convert_anno(annoinfo):
    clsname_map = {
        'ascus': 'ASC-US',
        'lsil': 'LSIL',
        'asch': 'ASC-H',
        'hsil': 'HSIL',
        'scc': 'HSIL',
        'agc': 'AGC',
        'trichomonas': 'NILM',
        'candida': 'NILM',
        'flora': 'NILM',
        'herps': 'NILM',
        'actinomyces': 'NILM',
    }
    categories_names = [item['name'] for item in annoinfo['categories']]
    anno_img = {}

    for item in annoinfo['annotations']:
        x,y,w,h = item['bbox']
        sub_class = clsname_map[categories_names[item['category_id']-1]]
        annoitem = {
            'id': item['id'],
            'sub_class': sub_class, 
            'region':{'x':x,'y':y,'width':w,'height':h}
        }
        if item['image_id'] in anno_img.keys():
            if sub_class in POSITIVE_CLASS:
                anno_img[item['image_id']].append(annoitem)
        else:
            anno_img[item['image_id']] = []
            if sub_class in POSITIVE_CLASS:
                anno_img[item['image_id']].append(annoitem)
    
    return anno_img
    
def gene_coco_json():
    annojson_save_dir = f'{root_dir}/annofiles'

    for mode in ['train', 'test']:
    # for mode in ['train']:
        image_dir = f'{root_dir}/{mode}'
        json_path = f'{root_dir}/{mode}.json'
        with open(json_path, 'r') as f:
            annoinfo = json.load(f)
        anno_img = convert_anno(annoinfo)

        coco_anno = {
            'images': [],
            'annotations': [],
            'categories': [{'id':idx+1, 'name':name, 'supercategory': ''} for idx,name in enumerate(POSITIVE_CLASS)],

        }
        imgid,annid = 1,1
        for fidx, filename in enumerate(tqdm(os.listdir(image_dir))):
            # if fidx > 10:
            #     break
            img = cv2.imread(f'{image_dir}/{filename}')
            h,w,_ = img.shape
            inside_items = anno_img[filename]
            purename = filename.split('.')[0]

            cut_points = generate_cut_regions((0,0), w, h, WINDOW_SIZE, STRIDE)
            for iidx,rect_coords in enumerate(cut_points):
                x1,y1 = rect_coords
                x2,y2 = x1+WINDOW_SIZE,y1+WINDOW_SIZE
                item_inside_patch = get_cutregion_inside([x1,y1,x2,y2], inside_items)
                filename = f'{purename}_{iidx}.png'
                imgitem = {
                    "id": imgid,
                    "width": WINDOW_SIZE,
                    "height": WINDOW_SIZE
                }
                
                if len(item_inside_patch) == 0:
                    imgitem['file_name'] = f'Neg/{filename}'
   
                else:
                    imgitem['file_name'] = f'Pos/{filename}'
                    for bboxitem in item_inside_patch:
                        clsid = classes.index(bboxitem['sub_class'])
                        bbox = [
                            bboxitem['region']['x'],
                            bboxitem['region']['y'],
                            bboxitem['region']['width'],
                            bboxitem['region']['height'],
                        ]
                        annitem = {
                            "id": annid,
                            "image_id": imgid,
                            "category_id": clsid,
                            "bbox": bbox,
                            "area": bbox[2]*bbox[3],
                            "iscrowd": 0
                        }
                        coco_anno['annotations'].append(annitem)
                        annid += 1

                coco_anno['images'].append(imgitem)
                imgid += 1
        
        with open(f'{annojson_save_dir}/OD_instances_{mode}.json', 'w') as f:
            json.dump(coco_anno, f)
        
def visual_sample():
    # 加载 COCO 标注文件
    annFile = f'{root_dir}/annofiles/OD_instances_train.json'
    coco = COCO(annFile)
    # 为每个类别生成一个颜色
    category_colors = {}
    categories = coco.loadCats(coco.getCatIds())
    for cat in categories:
        # 随机生成一个颜色 (R, G, B)
        color = (random.random(), random.random(), random.random())
        category_colors[cat['id']] = color

    # 获取所有图像 ID
    imgIds = coco.getImgIds()
    filtered_imgIds = [[],[],[]]
    for imgId in tqdm(imgIds,ncols=80):
        annIds = coco.getAnnIds(imgIds=imgId)
        anns = coco.loadAnns(annIds)
        
        # case:1 图像中的标注框只有一个且面积小于 64*64 的
        if len(anns) == 1 and len(filtered_imgIds[0])<5:
            bbox = anns[0]['bbox']
            area = bbox[2] * bbox[3]  # 宽 * 高
            if area < 64 * 64:
                filtered_imgIds[0].append(imgId)
                continue
        
        # case:2 图像中的标注框类别多于2种类别的
        if len(filtered_imgIds[1])<5:
            category_ids = [ann['category_id'] for ann in anns]
            unique_categories = set(category_ids)
            if len(unique_categories) > 2:
                filtered_imgIds[1].append(imgId)

        # case:3 图像中的标注框同时存在一个面积小于64*64的和一个面积大于200*200的
        if len(filtered_imgIds[2])<5:
            has_small,has_large = False,False
            for ann in anns:
                bbox = ann['bbox']
                area = bbox[2] * bbox[3]  # 宽 * 高
                if area < 64 * 64:
                    has_small = True
                if area > 200 * 200:
                    has_large = True
            if has_small and has_large:
                filtered_imgIds[2].append(imgId)
    # 可视化筛选出的图像
    for imgIds,dirname in zip(filtered_imgIds,['case1','case2','case3']):
        sample_save_dir = f'statistic_results/cdetector/{dirname}'
        os.makedirs(sample_save_dir, exist_ok=True)
        for imgId in imgIds:
            imgInfo = coco.loadImgs(imgId)[0]
            imgPath = f'{root_dir}/images/' + imgInfo['file_name']
            image = cv2.imread(imgPath)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            plt.imshow(image)
            ax = plt.gca()
            annIds = coco.getAnnIds(imgIds=imgId)
            anns = coco.loadAnns(annIds)
            for ann in anns:
                bbox = ann['bbox']
                category_id = ann['category_id']
                color = category_colors[category_id]
                x, y, w, h = bbox
                rect = Rectangle((x, y), w, h, linewidth=2, edgecolor=color, facecolor='none')
                ax.add_patch(rect)
                
                # 添加类别名称
                category_name = coco.loadCats(category_id)[0]['name']
                plt.text(x, y, category_name, color='white', backgroundcolor=color, fontsize=8)
            purename = os.path.basename(imgInfo['file_name'])
            plt.savefig(f'{sample_save_dir}/{purename}')
            plt.cla()
    

if __name__ == '__main__':
    # root_dir = '/x22201018/datasets/CervicalDatasets/ComparisonDetectorDataset'
    root_dir = '/c23030/zly/datasets/CervicalDatasets/ComparisonDetectorDataset'
    # root_dir = '/disk/medical_datasets/cervix/ComparisonDetectorDataset'
    
    gene_coco_json()
    # visual_sample()
