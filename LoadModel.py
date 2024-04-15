import torch
import torch.nn as nn
import torchvision
import numpy as np
import random
from Models.resnet import ResNet
from Models.resnetzip import resnet20
from torch.utils.data import Dataset, SubsetRandomSampler
from Models.vgg import vgg16
from dataloader import get_partition_dataloader
import os
def load_models(arch, num_classes, datasetname, device, wsize, checkpoint_folder, randominit="False", diffinit="False", width_multiplier = 8):
    models = []
    for m_idx in range(wsize):
        seed = m_idx
        if randominit=='True':
            model_path = None
        else:
            model_path = os.path.join(checkpoint_folder,"model_%d.pth"%m_idx)
        models.append(load_model(arch, num_classes, datasetname, width_multiplier, device, seed, diffinit, model_path))
    return models

def load_model(arch, num_classes, datasetname, width_multiplier, device, seed, diffinit="False", path = None):
    if diffinit=='True':
        torch.manual_seed(seed)
    if arch == 'resnet20':
        if datasetname == 'mnist':
            model = resnet20(channel=1, w = width_multiplier , num_classes = num_classes)
        else:
            model = resnet20(w = width_multiplier , num_classes = num_classes)
    elif arch == 'resnet50':
        model = torchvision.models.resnet50()
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes) 
    elif arch == 'vgg16':
        model = vgg16(num_classes = num_classes)
    if path != None:
        print("Loading pre-trained models.")
        model.load_state_dict(torch.load(path))
    else:
        if diffinit == 'True':
            print("Loading models with different random initialization")
        else:
            print("Loading models with the same random initialization")
    model.to(device)
        
    return model
        
        
        
        