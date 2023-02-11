import sys
sys.path.append('../img_ret')

import torch
from PIL import Image
from trainer import ImageRetrievalTrainer
from dataloaders.datasets import SOP, SOP_TEST
from model import TransformerEncoder

from torch.utils.data import DataLoader
from torchvision import transforms

# pip install faiss-gpu==1.6.3
# pip install transformers
# pip install wandb
# pip install dill

# pip install faiss
# pip install faiss-cpu
# pip install faiss-gpu==1.6.3
# pip uninstall faiss-cpu


# sys.executable('-m', 'pip', 'install', 'transformers')
import subprocess

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

    # #select pretrained resnet50 model for testing
    # model = Resnet50(embedding_size=512, pretrained=True, is_norm=1, bn_freeze = 1)

    trainer.test(test_loader)




if __name__ == "__main__":
    subprocess.check_call([sys.executable, "-m", "pip", "install", "transformers"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "faiss-gpu==1.6.3"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "wandb"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "dill"])


    # process output with an API in the subprocess module:
    reqs = subprocess.check_output([sys.executable, '-m', 'pip',
    'freeze'])
    installed_packages = [r.decode().split('==')[0] for r in reqs.split()]

    print(installed_packages)

    
    # img_ret()
    img_ret_test()