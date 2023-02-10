import sys
sys.path.append("../img_ret")
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from model import TransformerEncoder
from torchvision import transforms


class BaseDataset(Dataset):
    def __init__(self, transforms = 'None', split='train'):
        self.transforms = transforms
        self.split = split
        self.image_ids, self.class_ids, self.super_class_ids, self.im_paths = [],[],[],[]
        self.img_encoder = TransformerEncoder("swinv2", "base")

    def __len__(self,):
        return len(set(self.class_ids))*3
        # return 5000

    def __getitem__(self, idx):
        
        class_ids_numpy = np.array(self.class_ids)
        unique_class_ids = np.unique(class_ids_numpy) # to give equal weightage to all classes

        while (True): # loop untill we get proper triplets

            # anchor sample
            anchor_class_id = np.random.choice(unique_class_ids, replace=True) # picking an anchor class at random

            # positive
            # finding all indexes for the selected class_id using numpy
            anchor_class_indexes = np.where(class_ids_numpy==anchor_class_id)[0]

            if len(anchor_class_indexes) < 2:
                continue

            anchor_positive_idx = np.random.choice(anchor_class_indexes, size=2, replace=False)
            anchor_image, positive_image = self.im_paths[anchor_positive_idx[0]], self.im_paths[anchor_positive_idx[1]] 
            
            # negative sample
            remaining_ids = np.delete(class_ids_numpy, np.where(class_ids_numpy == anchor_class_id))
            negative_class_idx = np.random.choice(remaining_ids, replace=True) # picking an anchor class at random
            negative_image = self.im_paths[negative_class_idx]

            break

        return [
                self._get_transformed_image(anchor_image), \
                (self._get_transformed_image(positive_image),  torch.tensor(1)), \
                (self._get_transformed_image(negative_image),  torch.tensor(-1))
                ]

    def _get_transformed_image(self, im_path):
        img = Image.open(im_path)

        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')

        img = self.transforms(img)
        image = self.img_encoder.image_processor(img, return_tensors="pt")
        return image



class SOP(BaseDataset):
    def __init__(self, transforms, split='train'):
        self.image_root = '/ds/images/Stanford_Online_Products/'
        self.train_label = '/ds/images/Stanford_Online_Products/Ebay_train.txt'
        self.test_label = '/ds/images/Stanford_Online_Products/Ebay_test.txt'

        self.transforms = transforms
        self.split = split
        self.classes = range(0,22634)

        super(SOP, self).__init__(self.transforms,)

        # metadata = open(os.path.join(self.root, 'Ebay_train.txt' if self.classes == range(0, 11318) else 'Ebay_test.txt'))
        metadata = open(self.train_label)
        for i, (image_id, class_id, super_id, path) in enumerate(map(str.split, metadata)):
            if i > 0:
                if int(class_id)-1 in self.classes:
                    self.image_ids.append(int(image_id)-1)
                    self.class_ids.append(int(class_id)-1)
                    self.super_class_ids.append(int(super_id)-1)
                    self.im_paths.append(os.path.join(self.image_root, path))



# transform:

# dont convert to tensor in transform
train_transforms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])