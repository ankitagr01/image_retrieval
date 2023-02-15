import os
import sys
sys.path.append('../img_ret')

import torch
import torch.nn as nn

import pickle
import dill as pickle
from tqdm import tqdm

from torch.optim import lr_scheduler
from torch.utils import data as torch_data
from transformers import Swinv2Model

# Faiss
import faiss

# Import TripletMarginLoss & CosineDistance
# from pytorch_metric_learning import losses

from typing import Union, Tuple

# Track training progress using wandb
import wandb
wandb.login()

import time

# Disable wandb sync
os.environ['WANDB_MODE'] = 'dryrun'


## adding logger for debugging
import logging as log
root_logger = log.getLogger()
root_logger.setLevel(log.DEBUG)
handler = log.FileHandler('debug_model.log', 'w', 'utf-8')
handler.setFormatter(log.Formatter('%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s'))
root_logger.addHandler(handler)

""" 
This file provides methods to finetune & validate Image Retrieval pipelines. 
"""

class ImageRetrievalTrainer(object):
    def __init__(self, model:Union[Swinv2Model, None],
                epochs:int, 
                learning_rate:float,
                device:str) -> None:
        
        self.model = model
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.device = device

        # Init the wandb project
        wandb.init(project='img_retrieval', entity='ankit_dfki',
                    config={'epochs': self.epochs,
                            'learning_rate': self.learning_rate})

        # Init the AdamW optimizer function
        self.optimizer = torch.optim.AdamW(self.model.parameters(),
                                        lr=self.learning_rate,
                                        eps=1e-8,
                                        weight_decay=0.01,
                                        amsgrad=True)

        # Scheduler with cosine annealing scheme
        self.scheduler = lr_scheduler.CosineAnnealingLR(self.optimizer,
                                                        T_max=self.epochs,
                                                        eta_min=1e-8)

        # Init TipletMarginLoss with Euclidean Distance
        # self.triplet_loss = losses.TripletMarginLoss(margin=0.2)

        # Track the gradients of the model
        wandb.watch(self.model, log='all')

        # Move the model to the device
        self.model.to(self.device)

        # Total number of trainable parameters in the model
        print(f'Total number of trainable parameters in the model: {self.count_parameters()}')


    def count_parameters(self) -> int:
        """ Count the total number of parameters in the model. """
        return sum(p.numel() for p in self.model.parameters() 
                    if p.requires_grad)
                                            

    def train(self, train_dataloader:torch_data.DataLoader, 
                    val_dataloader:torch_data.DataLoader) -> None:
        """ Finetune the model using metric learning loss. """

        if train_dataloader is None or val_dataloader is None:
            raise ValueError('Either of train_dataloader and val_dataloader cannot be None.')

        train_start_time = time.time()
        print(f'\nFinetuning the model for {self.epochs} epochs...')

        with tqdm(total=self.epochs, desc='Epochs') as pbar:
            for epoch in range(self.epochs):
                # compute epoch time
                start_time = time.time()

                # Training
                train_loss = self.train_epoch(train_dataloader)

                # Update the LR Scheduler
                self.scheduler.step()

                # Validation
                val_loss = self.validate(val_dataloader)

                # Print the loss values
                print(f'Epoch: {epoch+1}/{self.epochs}, Train Loss: {train_loss}, Val Loss: {val_loss}')

                pbar.update(1)

                # save_path = '../models/'
                save_path = 'models_ckpt/'
                
                # time taken for each epoch
                epoch_time = time.time() - start_time
                print(f'Time taken for epoch {epoch+1} is {epoch_time} seconds.')
                
                # Save the model
                self.save_model(save_path, epoch=epoch,)

        # Finish the wandb run
        wandb.finish()

        # Total time taken for training
        total_time = time.time() - train_start_time
        print(f'Total time taken for training is {total_time} seconds.')


    def train_epoch(self, train_dataloader:torch_data.DataLoader) -> float:
        """ Train the model for one epoch. """
        self.model.train()

        total_loss = 0
        for batch in train_dataloader:

            # Move the data to the device
            anchor = batch[0]['pixel_values'].to(self.device, 
                                            dtype=torch.float).squeeze(1)
            
            pos = batch[1][0]['pixel_values'].to(self.device, 
                                            dtype=torch.float).squeeze(1)
            
            neg = batch[2][0]['pixel_values'].to(self.device,
                                            dtype=torch.float).squeeze(1)


            pos_label = batch[1][1].to(self.device, dtype=torch.float)
            neg_label = batch[2][1].to(self.device, dtype=torch.float)

            anchor_embedding = self.model(anchor)['pooler_output']
            pos_embedding = self.model(pos)['pooler_output']
            neg_embedding = self.model(neg)['pooler_output']

            # Compute triplet margin loss from pytorch library
            loss = nn.TripletMarginLoss(margin=1)(anchor_embedding,
                                                    pos_embedding,
                                                    neg_embedding)

            self.optimizer.zero_grad()

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

            # Track all the losses using wandb
            wandb.log({'total_train_loss': loss.item()})

        return total_loss / len(train_dataloader)


    def validate(self, val_dataloader:torch_data.DataLoader) -> float:
        """ Validate the model. """
        self.model.eval()

        total_loss = 0
        with torch.no_grad():
            for batch in val_dataloader:
                
                # Move the data to the device
                anchor = batch[0]['pixel_values'].to(self.device, 
                                                dtype=torch.float).squeeze(1)
                
                pos = batch[1][0]['pixel_values'].to(self.device, 
                                                dtype=torch.float).squeeze(1)
                
                neg = batch[2][0]['pixel_values'].to(self.device,
                                                dtype=torch.float).squeeze(1)

                pos_label = batch[1][1].to(self.device, dtype=torch.float)
                neg_label = batch[2][1].to(self.device, dtype=torch.float)

                anchor_embedding = self.model(anchor)['pooler_output']
                pos_embedding = self.model(pos)['pooler_output']
                neg_embedding = self.model(neg)['pooler_output']
                
                # Compute triplet margin loss from pytorch library
                loss = nn.TripletMarginLoss(margin=1)(anchor_embedding,
                                                        pos_embedding,         neg_embedding) 
                                                        
                total_loss += loss.item()

                # Track all the losses using wandb
                wandb.log({'total_val_loss': loss.item()})
        return total_loss / len(val_dataloader)

    
    def test(self, test_dataloader:torch_data.DataLoader) -> None:
        start_total = time.time()
        """ Test the model. """
        print(f'\nTesting the model...')

        # load model from stae_dict
        self.model.load_state_dict(torch.load('models_ckpt/model_v1_epoch_15'))
        self.model.eval()

        # Get the embeddings for the test set
        test_embeddings = []
        test_labels = []
        with torch.no_grad():
            for batch in test_dataloader:
                # Move the data to the device
                anchor = batch[0]['pixel_values'].to(self.device, 
                                                dtype=torch.float).squeeze(1)
                anchor_embedding = self.model(anchor)['pooler_output']
                test_embeddings.append(anchor_embedding)
                test_labels.append(batch[1])

        test_embeddings = torch.cat(test_embeddings, dim=0)
        test_labels = torch.cat(test_labels, dim=0)

        print('test_embeddings.shape: ', test_embeddings.shape)

        # self.model.to('cpu')

        # Compute the recall and MAP
        recall_1, recall_10, recall_100, recall_1000, map_ = self.compute_recall_map(test_embeddings, test_labels)

        print(f'Recall@1: {recall_1}, Recall@10: {recall_10}, Recall@100: {recall_100}, Recall@1000: {recall_1000}, MAP: {map_}')

        print(f'Total time taken for testing: {time.time() - start_total} seconds')	

    # Compute the recall and MAP using Facebook's Faiss library in GPU
    def compute_recall_map(self, test_embeddings:torch.Tensor,
                            test_labels:torch.Tensor) -> Tuple[float, float, float, float, float]:
        """ Compute the recall and MAP. """
        # Compute the recall and MAP
        recall_1, recall_10, recall_100, recall_1000, map_ = 0, 0, 0, 0, 0

        
        print('inside faiss')
        log.debug('****inside faiss')

        faiss_index_time = time.time()

        # Create a Faiss index
        index = faiss.IndexFlatL2(test_embeddings.shape[1]) # L2 distance
        print('index_cpu', index)
        log.debug(f'****index_cpu: {index}')

        # Get the number of GPUs
        ngpus = faiss.get_num_gpus()
        print("number of GPUs:", ngpus)
        log.debug(f'****number of GPUs: {ngpus}')

        # Make the index into a GPU index
        index = faiss.index_cpu_to_all_gpus(index) # make it a GPU index
        print('index_gpu', index)
        log.debug(f'****index_gpu: {index}')

        print(f'Total time taken to make faiss index: {time.time() - faiss_index_time} seconds')
        faiss_add_time = time.time()

        # Add the test embeddings to the index
        index.add(test_embeddings.cpu().numpy())

        print(f'Total time taken to add in faiss index: {time.time() - faiss_add_time} seconds')
        faiss_search_time = time.time()

        print('index total length', index.ntotal)
        log.debug(f'****index total length: {index.ntotal}')

        # Get the top 1000 nearest neighbors for each test embedding
        D, I = index.search(test_embeddings.cpu().numpy(), 1001)

        print(f'Total time taken to search in faiss index: {time.time() - faiss_search_time} seconds')
        print(I)
        recall_start_time = time.time()

        # Compute the recall and MAP
        for i in range(test_embeddings.shape[0]):
            # Get the top 1000 nearest neighbors for the ith test embedding
            neighbors = I[i]
            
            # Get the labels of the nearest neighbors
            neighbor_labels = test_labels[neighbors]

            # Get the label of the ith test embedding
            label = test_labels[i]

            # TODO: create a seperate function for recall@k calculation
            # Compute the recall and MAP
            if label in neighbor_labels[1:2]:
                recall_1 += 1
            if label in neighbor_labels[1:11]:
                recall_10 += 1
            if label in neighbor_labels[1:101]:
                recall_100 += 1
            if label in neighbor_labels[1:1001]:
                recall_1000 += 1

            # Compute the MAP
            for j in range(1001):
                if label in neighbor_labels[1:j+1]:
                    map_ += 1 / (j+1)
                    break
        
        # Compute the recall and MAP
        recall_1 /= test_embeddings.shape[0]
        recall_10 /= test_embeddings.shape[0]
        recall_100 /= test_embeddings.shape[0]
        recall_1000 /= test_embeddings.shape[0]
        map_ /= test_embeddings.shape[0]

        # return recall_1, recall_10
        print(f'Total time taken to compute recall and MAP: {time.time() - recall_start_time} seconds')
        
        log.debug(f'****recalls1: {recall_1}, recalls10: {recall_10}, recalls100: {recall_100}, recalls1000: {recall_1000}, map: {map_}')
        return recall_1, recall_10, recall_100, recall_1000, map_


    def save_model(self, save_path:str=None, epoch:int=None,
                    ) -> None:
        """ Save the finetuned model & its weights. """
        
        if save_path is None:
            raise ValueError('save_path cannot be None.')

        if not os.path.exists(save_path):
            os.makedirs(save_path)
        model_name = f'model_v1_epoch_{epoch+1}'

        # Save the model weights and optimizer state (recommended)
        save_dir = os.path.join(save_path, model_name)
        torch.save(self.model.state_dict(), save_dir)
        # torch.save({'state_dict': self.model.state_dict(), 'optimizer': self.optimizer.state_dict()}, save_dir)
        
        print(f"Model is saved in: {save_path} {model_name}\n")