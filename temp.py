# from pytorch_wavelets import DTCWTForward, DTCWTInverse
# import torch

# xfm = DTCWTForward(J=1, biort='near_sym_b', qshift='qshift_b')
# ifm = DTCWTInverse(biort='near_sym_b', qshift='qshift_b')

# x = torch.randn(3,1280,64,64)
# xl,xh = xfm(x)
# print()

import json
from collections import Counter
from cerwsi.utils import generate_cut_regions



cut_points = generate_cut_regions((0,0), 1650,1650, 800, 750, minlen=100)
print(len(cut_points))

