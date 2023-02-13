# image_retrieval

Part 1: Adaptation of exisiting models for Image retrieval task.
Part 2: Finetuning above model for image retrieval.
Part 3: Performance comparision for both the models. 

Results:
__Stanford Online Products__

Architecture | Training Type     | Loss     | Recall@1   | Recall@10  | Recall@100 | Recall@1000| MAP  
-------------|-------------------|----------|------------|------------|------------|------------|------
Swinv2Model  |  Untrained        |  Triplet |   17.89    |   29.65    |   46.42    |   69.87    | 12.1     
Swinv2Model  |  Pretrained       |  Triplet |   52.81    |   67.14    |   80.06    |   91.10    | 30.18     
Swinv2Model  |Finetuned(Epoch:1) |  Triplet |   69.91    |   83.02    |   91.36    |   96.78    | 38.38     

Last Edit: 13-Feb-2023

