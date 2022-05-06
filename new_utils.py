"""
Azad Amitoz 2021-08-30 21:10
"""


import torch
import numpy as np
from collections import Counter


def gen_mask(labels, seed_size =0, train_size=10, val_size=10, classes=10):
    """
    Hmm, I can do better than this.
    """
    new_seed_mask = torch.zeros(len(labels), dtype=torch.bool) #20
    new_train_mask = torch.zeros(len(labels), dtype=torch.bool) #20
    new_val_mask = torch.zeros(len(labels), dtype=torch.bool) #80
    for i in range(0,classes):
        full_mask = (labels == i)
        ids = torch.where(full_mask == True)[0]
        #train_mask = ids[torch.randperm(len(ids))[:20]]
        indices = ids[torch.randperm(len(ids))[:train_size+val_size+seed_size]]
        new_seed_mask[indices[:seed_size]] = True
        new_train_mask[indices[seed_size:seed_size+train_size]] = True
        new_val_mask[indices[seed_size+train_size:seed_size+train_size+val_size]] = True
    new_test_mask = ~(new_seed_mask + new_train_mask + new_val_mask)
    assert (torch.sum(new_test_mask) + torch.sum(new_train_mask) + 
    torch.sum(new_val_mask) + torch.sum(new_seed_mask) == len(labels))
    return new_seed_mask, new_train_mask, new_val_mask, new_test_mask 


def gen_mask_p(labels, seed_size_p=0, train_size_p=10, val_size_p=10, classes=10):
    """
    Hmm, I can do better than this.
    """
    new_seed_mask = torch.zeros(len(labels), dtype=torch.bool) #20
    new_train_mask = torch.zeros(len(labels), dtype=torch.bool) #20
    new_val_mask = torch.zeros(len(labels), dtype=torch.bool) #80
    count = Counter(labels.tolist())
    for i in range(0,classes):
        full_mask = (labels == i)
        ids = torch.where(full_mask == True)[0]
        #train_mask = ids[torch.randperm(len(ids))[:20]]

        seed_size = (seed_size_p * count[i])//100
        train_size = (train_size_p * count[i])//100
        val_size = (val_size_p * count[i])//100

        indices = ids[torch.randperm(len(ids))[:train_size+val_size+seed_size]]
        new_seed_mask[indices[:seed_size]] = True
        new_train_mask[indices[seed_size:seed_size+train_size]] = True
        new_val_mask[indices[seed_size+train_size:seed_size+train_size+val_size]] = True
    new_test_mask = ~(new_seed_mask + new_train_mask + new_val_mask)
    assert (torch.sum(new_test_mask) + torch.sum(new_train_mask) + 
    torch.sum(new_val_mask) + torch.sum(new_seed_mask) == len(labels))
    return new_seed_mask, new_train_mask, new_val_mask, new_test_mask 


def count_elements():
    """
    Get the number of elements in each class
    use the python counter class.
    """


def get_front(labels, seed_mask):
    new_front = []
    for i in range(0, labels.max()+1):
        mask = (labels == i)
        seeds = torch.where((mask*seed_mask) == True,1,0)
        new_front.append(seeds)
    Front = torch.vstack(new_front).type(torch.float32)
    return Front
