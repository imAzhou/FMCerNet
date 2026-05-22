import os.path as osp
from PIL import Image
from openslide import OpenSlide
from kfbreader import KFBSlide

def get_slide(wsi_path):
    '''获取切片对象'''
    ext = osp.splitext(wsi_path)[1].lower()
    if ext in ['.svs', '.tif', '.tiff', '.mrxs']:
        slide = OpenSlide(wsi_path)
    elif ext in ['.kfb']:
        slide = KFBSlide(wsi_path)
    else:
        raise ValueError(f'Unsupport extension: {wsi_path}')
    return slide

def read_region(slide, location, level, size, zero_level_loc=True) -> Image:
    '''
    读取切片指定层级的指定区域
    slide: get_slide函数获取的切片对象
    location: 要读取区域的左上角坐标(x, y)
    level: 要读取的缩放层级
    size: 要读取的区域图片大小
    zero_level_loc: 若为True，则location参数为左上角在level 0上的坐标，否则location为当前level上的左上角坐标
    '''
    ratio = slide.level_downsamples[level] / slide.level_downsamples[0]
    if isinstance(slide, KFBSlide):
        if zero_level_loc:
            return Image.fromarray(slide.read_region((round(location[0]/ratio), round(location[1]/ratio)), level, size))
        return Image.fromarray(slide.read_region(location, level, size))
    elif isinstance(slide, OpenSlide):
        if zero_level_loc:
            return slide.read_region(location, level, size)
        return slide.read_region((round(location[0]*ratio), round(location[1]*ratio)), level, size)
    else:
        raise ValueError(f'Unsupport slide: {type(slide)}')