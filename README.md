# image_retrieval
#### Ankit Agrawal (s8anagra@stud.uni-saarland.de) (ankitagr.nitb@gmail.com)


## 1. Overview
This repository contains a pipeline to:
1. Adapt existing model (Swinv2model) for Image retrieval task. 
2. Finetune above model for image retrieval (deep metric learning). 
3. Evaluation of above models for image retrieval task. 


__Architectures__
* Swin Transformer V2 model (https://arxiv.org/abs/2111.09883)
  
  
__Loss Functions__
* Triplet Loss (https://arxiv.org/abs/1412.6622)


__Sampling Methods__
* Random Sampling


__Datasets__
* Stanford Online Products (http://cvgl.stanford.edu/projects/lifted_struct/)


## 2. Repo & Dataset Structure
### 2.1 Repo Structure
```
Repository
│   ### General Files
│   README.md
│   requirements.txt    
│   run.sh
|
|   ### Main Scripts
|   main.py     (main driver code for training and evaluating)
|   trainer.py   (train, valid, test pipeline)
│   
│ 
│   ### Network Scripts
|   model.py    (contains implementation for Swinv2 transformer model)
│   
│    
└───dataloaders (should be added, if one does not want to set paths)
|    │   datasets.py (dataloaders for SOP datasets (train, val, test split))
|    │   stanford_online_products
```

### 2.2 Dataset Structures
__Stanford Online Products__
```
stanford_online_products
| Ebay_train.txt (0:9000 Train images, 9000:11318 Validation images *Shuffled set*)
| Ebay_test.txt (11318:22634 Test images)
```


## 3. Implementation

1. We use Stanford Online Products (SOP) dataset and use the standard train-test split for image retrieval task.
2. We use pretrained Swinv2 Transformer base model and adapt it to image retrieval task. 
3. We finetune Swinv2model using the train-val split of SOP dataset using triplet margin loss. 
└───For each image(anchor), we pick one image of same class (positive pair), and one image from different class (negative pair), forming a triplet sample of (anchor, positive, negative) image.
└───Total of 9000\*5 = 45000 triplet samples for training and 11590 triplet samples for validation
4. We use triplet margin loss as the loss function
5. Batch size= 16, training epochs = 15 (driven by validation set)
6. Training performed in single A100 40GB GPU.
7. Weights and Baises(wandb) for monitoring

To run the experiments:
1. Install packages listed in requirements.txt
2. Download SOP dataset to the Dataloaders folder and check for the paths in datasets.py
3. Update the wandb username details for monitoring
4. run main.py for training/evaluation
5. Training models checkpoints will be saved in models_ckpt directory


## 4. Results
Evaluation on the test set for Swinv2model and Finetuned Swinv2model on SOP dataset is presented below:
We compare Recall@k (k = 1,10,100,1000) along with MAP for the SOP evaluation standard. 

__Stanford Online Products__

Architecture | Training Type     | Loss     | Recall@1   | Recall@10  | Recall@100 | Recall@1000| MAP  
-------------|-------------------|----------|------------|------------|------------|------------|------
Swinv2Model  |  Untrained        |  Triplet |   17.89    |   29.65    |   46.42    |   69.87    | 12.1     
Swinv2Model  |  Pretrained       |  Triplet |   52.81    |   67.14    |   80.06    |   91.10    | 30.18     
Swinv2Model  |Finetuned(Epoch:1) |  Triplet |   69.91    |   83.02    |   91.36    |   96.78    | 38.38  
Swinv2Model  |Finetuned(Epoch:6) |  Triplet |   72.35    |   85.10    |   92.68    |   97.17    | 39.50  
Swinv2Model  |Finetuned(Epoch:10)|  Triplet |   __75.22__    |   __87.39__    |   __93.76__    |   __97.60__    | __40.78__

## 4. Model performance
Training time, inference time, 
Faiss
model size
notes on deployment
encoding dimensions


## 4. Tech-stack and Hardware

1. Python
2. Pytorch
3. Faiss: for embeddings similarity computation
4. A100-40GB GPU for training and inference
5. Weights and Baises (wandb)
6. srun: slurm workload manager
7. Grafarna


## 5. Extension
use hard mining sampling
triplet loss v/s contrastive loss for representation learning







Last Edit: 15-Feb-2023




Part 1: Adaptation of exisiting models for Image retrieval task.
Part 2: Finetuning above model for image retrieval.
Part 3: Performance comparision for both the models. 
