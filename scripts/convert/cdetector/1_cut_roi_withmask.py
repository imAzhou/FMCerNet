import os
from PIL import Image
import json
from scipy import sparse
from pycocotools import mask as mask_utils
from cerwsi.utils import generate_cut_regions
from tqdm import tqdm
import numpy as np
import warnings
from scipy.ndimage import find_objects
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pycocotools.coco import COCO

os.environ['CUDA_VISIBLE_DEVICES'] = '2'
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.modules.conv")


WINDOW_SIZE = 400
STRIDE = WINDOW_SIZE - 50
POSITIVE_CLASS = ['AGC', 'ASC-US','LSIL', 'ASC-H', 'HSIL', 'SCC']
CLASS_COLORS = [[31,119,180], [255,153,153], [255,105,180], [255,20,147], [139,0,139],[106,0,50]]
clsname_map = {
    'ascus': 'ASC-US',
    'lsil': 'LSIL',
    'asch': 'ASC-H',
    'hsil': 'HSIL',
    'scc': 'SCC',
    'agc': 'AGC',
    'trichomonas': 'NILM',
    'candida': 'NILM',
    'flora': 'NILM',
    'herps': 'NILM',
    'actinomyces': 'NILM',
}

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image) 

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))  

def vis_patch_sample(image, bboxes, clsnames, patch_mask, filename):
    plt.figure(figsize=(6, 6))
    plt.imshow(image)

    ax = plt.gca()
    for bbox,clsname,annmask in zip(bboxes,clsnames,patch_mask):
        bx1, by1, bx2, by2 = bbox
        bw,bh = bx2-bx1, by2-by1
        ax.add_patch(patches.Rectangle((bx1, by1), bw, bh, edgecolor='red', facecolor='none', linewidth=2))
        ax.text(
            bx1, by1 - 4,  # 往上偏移一点
            clsname,
            fontsize=8,
            color='red',
            verticalalignment='top',
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=1)
        )
        show_mask(annmask, ax, random_color=True)
        
    plt.axis('off')
    plt.tight_layout()
    os.makedirs('statistic_results/CDetector/patch_from_RoI', exist_ok=True, mode=0o777)
    plt.savefig(f'statistic_results/CDetector/patch_from_RoI/{filename}')
    plt.close()

def cut_img(roi_img, patch_coords):
    patch = Image.new('RGB', (WINDOW_SIZE, WINDOW_SIZE), color=(255, 255, 255))
    rw, rh = roi_img.size
    x1,y1,x2,y2 = patch_coords
    # 计算与原图交集区域
    int_x1 = max(0, x1)
    int_y1 = max(0, y1)
    int_x2 = min(x2, rw)
    int_y2 = min(y2, rh)
    # 从原图裁剪交集区域
    cropped = roi_img.crop((int_x1, int_y1, int_x2, int_y2))
    # 计算粘贴位置（相对于 patch 左上角）
    paste_x = int_x1 - x1
    paste_y = int_y1 - y1
    # 粘贴到 patch 中
    patch.paste(cropped, (paste_x, paste_y))
    
    return patch

def gene_patch_jsonlist(npz_mask_save_dir, img_save_dir, json_save_dir):
    for mode in ['train', 'test']:
        jsonfile = f'{root_dir}/{mode}.json'
        with open(jsonfile, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        coco = COCO(jsonfile)
        
        format_result = {
            'info': {},
            'categories': [{
                'id': idx+1,
                'name': clsname,
                'color': clscolor,
            } for idx, clsname,clscolor in zip(range(len(POSITIVE_CLASS)), POSITIVE_CLASS, CLASS_COLORS)],
            'images': [],
            'annotations': []
        }
        imgid, annid = 0,0
        for imgitem in tqdm(json_data['images'], ncols=80):
            annids = coco.getAnnIds([imgitem['id']])
            annos = coco.loadAnns(annids)
            imgitem['annos'] = []
            for ann in annos:
                catinfo = coco.loadCats([ann['category_id']])[0]
                clsname = clsname_map[catinfo['name']]
                if ann['bbox'][2]>5 and ann['bbox'][3]>5 and clsname in POSITIVE_CLASS:
                    imgitem['annos'].append(ann)
            imgpath = f"{root_dir}/{mode}/{imgitem['file_name']}"
            purename = imgitem['file_name'].split('.')[0]
            roi_img = Image.open(imgpath)
            rw,rh = roi_img.size

            if len(imgitem['annos']) > 0:
                loader = np.load(f"{npz_mask_save_dir}/{purename}.npz")
                sparse_mask = sparse.coo_matrix((loader['data'], (loader['row'], loader['col'])), shape=loader['shape'])
                roi_mask = sparse_mask.toarray().astype(np.int16)
            else:
                roi_mask = np.zeros((rh,rw), dtype=np.int16)
            
            cut_points = generate_cut_regions((0,0), rw, rh, WINDOW_SIZE, STRIDE, minlen=100)
            for iidx,patch_coords in enumerate(cut_points):
                bboxes,clsids,patch_mask = calc_patch_anns(patch_coords, imgitem['annos'], roi_mask)
                
                filename = f'{purename}_{iidx}.png'
                diagnose = int(len(bboxes) > 0)
                prefix = 'Neg' if diagnose == 0 else 'Pos'
                
                cropimg = cut_img(roi_img, patch_coords)
                cropimg.save(f"{img_save_dir}/{prefix}/{filename}")

                # clsnames = [clsname_map[catinfo['name']] for catinfo in coco.loadCats(clsids)]
                # vis_patch_sample(cropimg, bboxes, clsnames, patch_mask, filename)

                format_result['images'].append(
                    {'id': imgid, 'width': WINDOW_SIZE, 'height': WINDOW_SIZE,
                    'file_name': f"{prefix}/{filename}", 
                    'extra_info': {
                            'prefix': prefix,
                            'square_coords': patch_coords
                        },
                    'diagnose': diagnose})
                for bbox,clsid,annmask in zip(bboxes,clsids,patch_mask):
                    bx1, by1, bx2, by2 = bbox
                    bw,bh = bx2-bx1, by2-by1
                    catinfo = coco.loadCats([clsid])[0]
                    clsname = clsname_map[catinfo['name']]
                    rle = mask_utils.encode(np.asfortranarray(annmask))
                    rle['counts'] = rle['counts'].decode('utf-8')
                    format_result['annotations'].append({
                        "id": annid,
                        "image_id": imgid,
                        "category_id": POSITIVE_CLASS.index(clsname) + 1,
                        "segmentation": rle,
                        "bbox": [bx1, by1,bw,bh],
                        "area": bw*bh,
                        "iscrowd": 0,
                    })
                    annid += 1
                imgid += 1
                
        with open(f'{json_save_dir}/{mode}.json', 'w', encoding='utf-8') as f:
            json.dump(format_result, f, ensure_ascii=False)

def calc_patch_anns(patch_coords, annlist, roi_mask):
    rpx1, rpy1, rpx2, rpy2 = patch_coords  # 相对 ROI 的坐标
    patch_mask = roi_mask[rpy1:rpy2, rpx1:rpx2]
    
    ann_bboxes, ann_clsids, new_patch_mask = [], [], []
    annidx = np.unique(patch_mask)
    if len(annidx) <= 1:
        return ann_bboxes, ann_clsids, new_patch_mask  # only background (0)

    # 使用 find_objects 获取所有非零实例的切片区域
    objects = find_objects(patch_mask)  # 返回一个 列表 slices，长度等于 input 中的最大 label 值（不包括 0）
    for aidx in annidx[1:]:  # 跳过背景 0
        obj_slice = objects[aidx-1]
        if obj_slice is None:
            continue
        yslice, xslice = obj_slice
  
        by1, by2 = yslice.start, yslice.stop
        bx1, bx2 = xslice.start, xslice.stop
        bwidth, bheight = bx2 - bx1, by2 - by1
        
        # 判断是否为贴边小目标
        is_small = min(bwidth, bheight) < 50
        is_near_edge = (
            bx1 <= 1 or by1 <= 1 or
            bx2 >= patch_mask.shape[1] - 1 or
            by2 >= patch_mask.shape[0] - 1
        )
        if is_small and is_near_edge:
            continue  # 丢弃该 annitem
        
        ann_bboxes.append([bx1, by1, bx2, by2])
        annitem = annlist[aidx - 1]
        ann_clsids.append(annitem['category_id'])
        new_patch_mask.append(patch_mask==aidx)

    return ann_bboxes, ann_clsids, new_patch_mask

def statistic_imgs():
    for mode in ['train','test']:
        with open(f'{json_save_dir}/{mode}.json', 'r', encoding='utf-8') as f:
            json_data = json.load(f)
    
        pn_cnt = [0,0]
        for imgitem in tqdm(json_data['images'], ncols=80):
            pn_cnt[imgitem["diagnose"]] += 1
        print(pn_cnt)

def vis_sample_img():
    for mode in ['train','test']:
        with open(f'{json_save_dir}/{mode}.json', 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        coco = COCO(f'{json_save_dir}/{mode}.json')
        
        for imgitem in tqdm(json_data['images'], ncols=80):
            annids = coco.getAnnIds([imgitem['id']])
            annos = coco.loadAnns(annids)
            img = Image.open(f"{img_save_dir}/{imgitem['file_name']}")
            vis_patch_sample(img, bboxes, clsnames, patch_mask, filename)

if __name__ == "__main__":
    root_dir = 'data_resource/ComparisonDetectorDataset'
    npz_mask_save_dir = f'{root_dir}/roi_inst_mask'
    img_save_dir = f'{root_dir}/WINDOW_SIZE_{WINDOW_SIZE}/images'
    os.makedirs(f"{img_save_dir}/Neg", exist_ok=True, mode=0o777)
    os.makedirs(f"{img_save_dir}/Pos", exist_ok=True, mode=0o777)
    json_save_dir = f'{root_dir}/WINDOW_SIZE_{WINDOW_SIZE}/annofiles'
    os.makedirs(json_save_dir, exist_ok=True, mode=0o777)
    gene_patch_jsonlist(npz_mask_save_dir,img_save_dir,json_save_dir)

    statistic_imgs()
    # vis_sample_img()

'''
WINDOW_SIZE = 400, STRIDE = 350:
train ['neg', 'pos']: [28702, 32532]
val ['neg', 'pos']: [3572, 3678]
'''
