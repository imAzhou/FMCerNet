# import os
# from cerwsi.utils import KFBSlide,kfbslide_get_associated_image_names,kfbslide_read_associated_image
# import openslide

# svs_path = '/nfs-medical3/data/浙一胃HE与Fish数据/胃癌FISH片/2025126768-FISH-HE-KL.svs'
# kfb_path = '/medical-data/data/cervix/JFSW_1109/HSIL/C202028855.kfb'
# kfbf_path = '/nfs-medical3/data/浙一胃HE与Fish数据/胃癌FISH片/2025126768-FISH-KL.kfbf'

# # source_path = kfbf_path
# # slide = KFBSlide(source_path)
# # swidth, sheight = slide.level_dimensions[0]
# # associated_images = kfbslide_get_associated_image_names(slide._osr)
# # if 'label' not in associated_images:
# #     print(f'{source_path} haven\'t label!')
# # else:
# #     filename = os.path.splitext(os.path.basename(source_path))[0]
# #     image = kfbslide_read_associated_image(slide._osr, 'label')
# #     output_path = f"{filename}.png"
# #     image.save(output_path, "PNG")

# slide = openslide.OpenSlide(svs_path)
# print(slide.associated_images.keys())


import torch

data = torch.load('data_resource/0630/WINDOW_SIZE_1200/slide_feat_detector/JFSW_1_2.pt')
print()