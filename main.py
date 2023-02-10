import sys
sys.path.append('../img_ret')

import torch
from PIL import Image
from trainer import ImageRetrievalTrainer
from dataloaders.datasets import SOP, SOP_TEST
from model import TransformerEncoder

from torch.utils.data import DataLoader
from torchvision import transforms

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

if 'cuda' not in device:
    raise ValueError('GPU not available. Please use a GPU to run this script.')

# dont convert to tensor in transform
train_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(),
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
])

def img_ret():
    train_sop_dataset = SOP(train_transforms, split='train')
    val_sop_dataset = SOP(val_transforms, split='val')

    train_loader = DataLoader(train_sop_dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_sop_dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)

    te_obj = TransformerEncoder(model_type="swinv2",
                                 model_size="base")

    trainer = ImageRetrievalTrainer(device=device,
                                  model=te_obj.model,
                                  epochs=15,
                                  learning_rate=1e-4,)

    trainer.train(train_loader, val_loader)


def img_ret_test():
    sop_dataset = SOP_TEST()

    test_loader = DataLoader(sop_dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)

    te_obj = TransformerEncoder(model_type="swinv2",
                                 model_size="base")

    trainer = ImageRetrievalTrainer(device=device,
                                  model=te_obj.model,
                                  epochs=3,
                                  learning_rate=1e-4,)

    trainer.test(test_loader)




if __name__ == "__main__":
    img_ret()
    # img_ret_test()