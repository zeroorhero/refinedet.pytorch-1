import torch
import numpy as np
from numpy import random
import pdb

random.seed(10)
# import random
batch = 2
height = 3
width = 4
num_priors = 2
len_box = 4

len = batch * num_priors * len_box * height * width
boxes = np.array(range(len)).reshape((batch, 
              height, width, num_priors*len_box))
print(boxes)
pytorch_boxes = np.transpose(boxes, (0, 3, 1, 2))
print(pytorch_boxes)

one_loc = np.transpose(pytorch_boxes, (0, 2, 3, 1))
print(one_loc)
assert np.all(boxes == one_loc), 'Equal'
                       
locs = [torch.Tensor(one_loc)] * 2

loc = torch.cat([o.view(o.size(0), -1) for o in locs], 1)
pdb.set_trace()
final_loc = loc.view(loc.size(0), -1, 4)
print(loc)

 

# boxes = (random.rand(batch, num_priors*len_box, 
#                      height, width) * 10).astype(np.int)


