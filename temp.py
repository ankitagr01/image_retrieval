import os
print('hellofg')
import numpy as np
import torch


# classes = [0, 1,2,3,4,5,6]
# ys, I, im_paths = [], [], []
# yss, II, im_pathss = [], [], []
# ll = [(1, 0, 'tef'), (123, 1, 'jejj'), (124, 1, 'jejj2'),  (127, 1, 'jejjsdgs2'), (128, 1, 'sdf'), (129, 1, 'fdgdf'), (125, 2, 'jfdhdfejj2'), (126, 3, 'jefdgdjj2')]
# for i, (image_id, class_id, path) in enumerate(ll):
#     if i > 0:
#         if int(class_id)-1 in classes:
#             ys += [int(class_id)]
#             I += [int(image_id)]
#             im_paths.append(os.path.join('root', path))

#             yss.append(int(class_id))
#             II.append(int(image_id))
#             im_pathss.append(os.path.join('root', path))


# print(ys)
# print(I)
# print(im_paths)
# print(yss)
# print(II)
# print(im_pathss)


#make a 1d tensor
ten = torch.tensor([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
print(ten)
print(ten.shape)
print(ten[:10])
print(ten[10:])
print(ten[10:15])
print(ten[1:2])
print(ten[1:11])
# print(ten[1:1])





# original_array = np.array([1, 2, 2, 4, 5, 7, 9, 12, 12])

# print(np.where(original_array == 1))
# print(np.where(original_array == 1)[0])
# print(np.where(original_array == 1)[1])
            