from collections import defaultdict
from collections.abc import MutableMapping
from typing import Sequence, Callable
import os
import math
import pdb
import torch
import torch.nn as nn
import numpy as np
import yaml
from tqdm.auto import tqdm
from copy import deepcopy
from sklearn.model_selection import train_test_split
from torch.cuda.amp import autocast, GradScaler
from torch.nn import CrossEntropyLoss
from torch.optim import SGD, Adam, lr_scheduler
from fvcore.nn.flop_count import flop_count
from inspect import getmembers, isfunction
from utils.metric_calculators import get_metric_fns
import torch.nn.functional as F
import clip
import einops
import torch
import scipy
import random
import string


CONCEPT_TASKS  = list(string.ascii_uppercase)

##########################################################################################################################
######################################################### CLASSES ########################################################
##########################################################################################################################

class SpaceInterceptor(nn.Module):
    '''
    This module is meant to intercept computational flows between any given two layers. 
    Inserting the module between two layers allows us to compute a merge/unmerge on each 
    layer separately, rather than a single merge/unmerge for both. This is most useful for
    controlling the transformations learned over residual connections. E.g., if we have a 
    case where we combine several residuals together, we can instead place this on each 
    branch before their connection, allowing us to learn distinct merge/unmerges on each
    branch, and 1 merge/unmerge on the connection, rather than 1 merge/unmerge for everything.
    Thus, it allows for (hopefully) more specificity.
    
    All it requires is a dimension parameter (the size of the feature dimension).
    
    It contains only 1 weight, which begins as the identity, and will be transformed according to
    the unmerge/merge that will be applied over it. For all intents and purposes, this is treated
    as a linear layer, with not bias! 
    '''
    def __init__(self, dim):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.eye(dim))
    
    def forward(self, input, kind='linear'):
        if kind == 'conv':
            input = input.permute(0, 2,3, 1)
        
        output = input @ self.weight.T
        
        if kind == 'conv':
            output = output.permute(0, 3, 1, 2)
        
        return output
    

def get_merging_fn(name):
    """ Get alignment function from name. """
    import matching_functions
    matching_fns = dict([(k, v) for (k, v) in getmembers(matching_functions, isfunction) if 'match_tensors' in k])
    return matching_fns[name]




# use the train loader with data augmentation as this gives better results
# taken from https://github.com/KellerJordan/REPAIR
def reset_bn_stats(model, loader, reset=True):
    """Reset batch norm stats if nn.BatchNorm2d present in the model."""
    device = get_device(model)
    has_bn = False
    # resetting stats to baseline first as below is necessary for stability
    for m in model.modules():
        if type(m) == nn.BatchNorm2d:
            if reset:
                m.momentum = None # use simple average
                m.reset_running_stats()
            has_bn = True

    """if not has_bn:
        return model"""

    # run a single train epoch with augmentations to recalc stats
    model.train()
    with torch.no_grad(), autocast():
        for images, _ in tqdm(loader, desc='Resetting batch norm'):
            _ = model(images.to(device))
    return model

def get_device(model):
    """Get the device of the model."""
    return next(iter(model.parameters())).device


