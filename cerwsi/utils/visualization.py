import numpy as np
import matplotlib.pyplot as plt


def show_mask(mask, ax, random_color=False, rgb=[30,144,255]):
    if rgb != [255,255,255]:
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            color = np.array([rgb[0]/255, rgb[1]/255, rgb[2]/255, 0.4])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_image)

def show_multi_mask(mask_multi_cls, ax, palette):
    for cls_i,rgb in enumerate(palette):
        mask = mask_multi_cls == cls_i
        show_mask(mask, ax, rgb=rgb)

def show_box(box, ax, edgecolor='green', min=0, max=1023):
    box = np.clip(np.array(box), min, max) 
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor=edgecolor, facecolor=(0,0,0,0), lw=2))
    