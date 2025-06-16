import numpy as np
from cellpose import models, core, io, plot, utils, transforms
from pathlib import Path
from tqdm import trange
import matplotlib.pyplot as plt
import cv2

io.logger_setup() # run this to get printing of progress

#Check if colab notebook instance has GPU access
if core.use_gpu()==False:
  raise ImportError("No GPU access, change your runtime")

model = models.CellposeModel(gpu=True)

# filename = "data_resource/cellpose/imgs_cyto3.npz"
# dat = np.load(filename, allow_pickle=True)["arr_0"].item()
# imgs = dat["imgs"]  # list of ndarray (2, imgh, imgw)
# masks_true = dat["masks_true"]  # list of ndarray (imgh, imgw)

# plt.figure(figsize=(8,3))
# for i, iex in enumerate([9, 16, 21]):
#     img = imgs[iex].squeeze()
#     plt.subplot(1,3,1+i)
#     plt.imshow(img[0], cmap="gray", vmin=0, vmax=1)
#     plt.axis('off')
# plt.tight_layout()
# plt.savefig('data_resource/cellpose/test.png')
img_url = 'data_resource/cellpose/JFSW_1_0_1211906498145_74.png'
img = cv2.imread(img_url)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
imgs = [img]
masks_pred, flows, styles = model.eval(imgs, 
                                       niter=1000, 
                                       max_size_fraction=1,
                                       diameter=7.5) # using more iterations for bacteria
nimg = len(imgs)
for idx in range(nimg):
    maski = masks_pred[idx]
    flowi = flows[idx][0]

    fig = plt.figure(figsize=(12,5))
    plot.show_segmentation(fig, imgs[idx], maski, flowi)
    plt.tight_layout()
    plt.savefig('data_resource/cellpose/infer_cervical.png')

# titles = [
#         "Cellpose", "Nuclei", "Tissuenet", "Livecell", "YeaZ",
#          "Omnipose\nphase-contrast", "Omnipose\nfluorescent",
#         "DeepBacs"
#     ]

# plt.figure(figsize=(12,6))
# ly = 400
# for iex in range(len(imgs)):
#     img = imgs[iex].squeeze().copy()
#     img = np.clip(transforms.normalize_img(img, axis=0), 0, 1) # normalize images across channel axis
#     ax = plt.subplot(3, 8, (iex%3)*8 + (iex//3) +1)
#     if img[1].sum()==0:
#         img = img[0]
#         ax.imshow(img, cmap="gray")
#     else:
#         # make RGB from 2 channel image
#         img = np.concatenate((np.zeros_like(img)[:1], img), axis=0).transpose(1,2,0)
#         ax.imshow(img)
#     ax.set_ylim([0, min(400, img.shape[0])])
#     ax.set_xlim([0, min(400, img.shape[1])])


#     # GROUND-TRUTH = PURPLE
#     # PREDICTED = YELLOW
#     outlines_gt = utils.outlines_list(masks_true[iex])
#     outlines_pred = utils.outlines_list(masks_pred[iex])
#     for o in outlines_gt:
#         plt.plot(o[:,0], o[:,1], color=[0.7,0.4,1], lw=0.5)
#     for o in outlines_pred:
#         plt.plot(o[:,0], o[:,1], color=[1,1,0.3], lw=0.75, ls="--")
#     plt.axis('off')

#     if iex%3 == 0:
#         ax.set_title(titles[iex//3])

# plt.tight_layout()
# plt.savefig('data_resource/cellpose/test_infer_result.png')