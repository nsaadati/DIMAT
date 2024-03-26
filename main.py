import torchvision.transforms as T
import torch.nn.functional as F
import os
import csv
import ast
import os
import pdb
import clip
import copy
import random
from time import time
import numpy as np
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from itertools import product
import argparse
import pandas as pd
import pickle
from train.train import train_model, test_accuracy
from utils.AM import *
from utils.model_merger import ModelMerge
from graphs.resnet_graph import resnet20 as resnet20_graph
from graphs.resnet_graph import resnet50 as resnet50_graph
from graphs.vgg_graph import vgg16 as vgg16_graph
from utils.matching_functions import match_tensors_zipit, match_tensors_optimal, match_tensors_permute, match_tensors_kmeans, match_tensors_randperm
from utils.metric_calculators import CovarianceMetric, MeanMetric, Py_CovarianceMetric, CorrelationMetric, CossimMetric
from utils.weight_matching import find_permutation, apply_permutation, resnet20_permutation_spec, resnet50_permutation_spec, vgg16_permutation_spec
from torch.utils.data import SubsetRandomSampler
from torch.utils.data import Subset
from data.dataloader import get_partition_dataloader
from data.dataset import Data_Loader
from Models.LoadModel import load_model, load_models

# Create argument parser
parser = argparse.ArgumentParser(description='Load models and calculate merged model')
parser.add_argument('--wsize', type=int, default=10, help='Number of agents')
parser.add_argument('--exp', type=int, default=1, help='graph indicator')
parser.add_argument('--data_dist', type=str, default='non-iid', choices=['iid', 'non-iid', 'non-iid-PF'], help='Degree of nonidness')
parser.add_argument('--num_epochs', type=int, default=2, help='Number of epochs for training')
parser.add_argument('--merg_itr', type=int, default=1, help='the iteration number for merging')
parser.add_argument('--merg_itr_init', type=int, default=0, help='the iteration number that strats with')
parser.add_argument('--ckp', type=str, default='checkpoint/gitreb', help='Path to the models checkpoint')
parser.add_argument('--save_folder', type=str, default='results', help='Folder path to save the models')
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument("--training", action="store_true" , help='train after merging or not')
parser.add_argument("--randominit", type=str, default="False" , help='start models with random initialization')
parser.add_argument("--diffinit", type=str, default="False" , help='start models with random and different initialization')
parser.add_argument('--phase',type=str, default='finetune', choices=('pretrain', 'finetune'), help='Define the data distribution')
parser.add_argument("--zerotorandom", action="store_true", help="change zero value in covrinse dignoal to random valueclose to epsilun")
parser.add_argument("--subdata", action="store_true", help="Calculate batch normalization for ZipIt using a subset of data")
parser.add_argument("--numdata", type=int, default=1000, help="Number of data points to use in the subset")
parser.add_argument('--optim', type=str, default='adam')
parser.add_argument('--num_classes', type=int, default=100, help='Number of the classes')
parser.add_argument('--dataset', type=str, default='cifar100', choices=['cifar100', 'tinyimgnt', 'cifar10', 'mnist'])
parser.add_argument('--match', type=str, default='permute', choices=['permute', 'kmeans', 'optimal', 'zipit', 'randperm'])
parser.add_argument('--matrix', type=str, default='cov', choices=['pycov', 'corr', 'cov', 'cossi'])
parser.add_argument('--imgntdir', type=str, default='data/tiny-imagenet-200', help="directory for tiny imagenet dataset")
parser.add_argument('--opt', type=str, default='DIMAT', help='the algorithm to merge models',
                   choices= ('WM', 'WA', 'DIMAT'))
parser.add_argument('--width-multiplier', type=int, default=8)
parser.add_argument('--model', type=str, default='resnet20', choices=['resnet20', 'resnet50', 'vgg16'],
                    help="Specify the architecture for models. Default is 'resnet20'.")
parser.add_argument('--seed', type=str, default='42',  help="Seed value for reproducibility")

# Parse the command-line arguments
args = parser.parse_args()

def seed_everything(random_seed):
    random.seed(random_seed)
    os.environ['PYTHONHASHSEED'] = str(random_seed)
    np.random.seed(int(random_seed))
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def init_model_state_dict(model_state_dict):
    for p in model_state_dict: 
        model_state_dict[p] = 0.0 * model_state_dict[p]
    return model_state_dict


def WA(models, cdict):
    merged_model_state_dicts = [init_model_state_dict(copy.deepcopy(models[0].state_dict())) for _ in range(len(models))]
    model_state_dicts = [model.state_dict() for model in models]
    for m_idx, model in enumerate(models):
        neighbors = [i for i, val in enumerate(cdict['connectivity'][m_idx]) if val > 0]
        pi_m_idx = cdict['pi'][m_idx]
        for neighbor_id in neighbors:
            for p in merged_model_state_dicts[m_idx]:
                merged_model_state_dicts[m_idx][p] +=  pi_m_idx[neighbor_id]*model_state_dicts[neighbor_id][p]
    merged_models = []
    for m_idx in range(len(models)):
        merged_models.append(copy.deepcopy(models[m_idx]))
        merged_models[m_idx].load_state_dict(merged_model_state_dicts[m_idx])
    return merged_models                  

def WM(models, cdict, perm_spec, device):
    merged_model_state_dicts = [init_model_state_dict(copy.deepcopy(models[0].state_dict())) for _ in range(len(models))]
    model_state_dicts = [model.state_dict() for model in models]
    for m_idx, model in enumerate(models):
        neighbors = [i for i, val in enumerate(cdict['connectivity'][m_idx]) if val > 0]
        pi_m_idx = cdict['pi'][m_idx]
        for l in merged_model_state_dicts[m_idx]:
            merged_model_state_dicts[m_idx][l] =  pi_m_idx[m_idx]*(models[m_idx].state_dict()[l])
        for neighbor_id in neighbors:
            if neighbor_id == m_idx:
                continue
            neighbor_model_state_dict=copy.deepcopy(models[neighbor_id].state_dict())
            for e in neighbor_model_state_dict:
                neighbor_model_state_dict[e] =  pi_m_idx[neighbor_id]*(models[neighbor_id].state_dict()[e])
            permutation_spec = perm_spec()
            final_permutation = find_permutation(permutation_spec, merged_model_state_dicts[m_idx], neighbor_model_state_dict)
            updated_params_neighbor = apply_permutation(permutation_spec, final_permutation, neighbor_model_state_dict)
            updated_params_neighbor = updated_params_neighbor
            for p in merged_model_state_dicts[m_idx]:
                merged_model_state_dicts[m_idx][p] += updated_params_neighbor[p].to(device)
    merged_models = []
    for m_idx in range(len(models)):
        merged_models.append(copy.deepcopy(models[m_idx]))
        merged_models[m_idx].load_state_dict(merged_model_state_dicts[m_idx])
    return merged_models 

def DIMAT(models, cdict, experiment, graph_func, device, num_classes, trainloader, testloader, match_func, metric_classes):

    exp = experiment
    merged_models = []  
    for m_idx, model in enumerate(models):
        interp_w = []
        print('Agent',m_idx)
        neighbors = [i for i, val in enumerate(cdict['connectivity'][m_idx]) if val > 0]
        pi_m_idx = cdict['pi'][m_idx]
        merged_models.append(copy.deepcopy(models[m_idx]))
        temp_models = []

        models[m_idx] = reset_bn_stats(models[m_idx], trainloader)
        temp_models.append(copy.deepcopy(models[m_idx]))
        interp_w.append(pi_m_idx[m_idx])
        for neighbor_id in neighbors:
            if neighbor_id == m_idx:
                continue
            print('Neighbor ID',neighbor_id)

            models[neighbor_id] = reset_bn_stats(models[neighbor_id], trainloader)
            temp_models.append(copy.deepcopy(models[neighbor_id]))
            interp_w.append(pi_m_idx[neighbor_id])

        graphs = [ graph_func(model).graphify() for model in temp_models]
        del temp_models
        Merge = ModelMerge(*graphs, device=device)
        Merge.transform(
            merged_models[m_idx], 
            trainloader, 
            transform_fn = match_func, 
            metric_classes = metric_classes,
            stop_at = None,
            interp_w = interp_w,
            **{'a': .0001, 'b': .075}
        )
        del graphs
        reset_bn_stats(Merge.to(device), trainloader)    
        New_Model_State_dict = copy.deepcopy(Merge.merged_model.state_dict())
        merged_models[m_idx].load_state_dict(New_Model_State_dict)
        
        if exp==1:
            for j in range(len(models)-1):
                merged_models.append(copy.deepcopy(merged_models[m_idx]))
            break
        del Merge
        del New_Model_State_dict
    return merged_models


def merge_models(models, cdict, opt, experiment, graph_func, device, num_classes, trainloader, testloader, match_func, metric_classes, perm_spec):
    if opt == 'WA':
        merged_models = WA(models, cdict)
    elif opt == 'WM':
        merged_models = WM(models, cdict, perm_spec, device)
    elif opt == 'DIMAT':
        merged_models = DIMAT(models, cdict, experiment, graph_func, device, num_classes, trainloader, testloader, match_func, metric_classes)
    return merged_models
def train_models(merged_models, training, wsize, num_epochs, batch_size, datasetname, data_dist, device, optimizer, trainset, phase):
    if not training:
        return                        
    for model_id, model in enumerate(merged_models):
        model_trainloader = get_partition_dataloader(trainset, data_dist, batch_size, wsize, datasetname, model_id, phase)
        train_model(model, num_epochs, train_loader=model_trainloader, device=device, optimizer_type=optimizer, softmax=True)
        

def write_accuracy_matrix_to_csv(output_file, accuracy_matrix, all_classes_accuracy, all_classes_loss):
    num_models = len(accuracy_matrix[0])
    accuracy_avg = np.mean(accuracy_matrix, axis=0)
    all_classes_accuracy_avg = np.mean(all_classes_accuracy)
    all_classes_loss_avg = np.mean(all_classes_loss)

    with open(output_file, 'w', newline='') as file:
        writer = csv.writer(file)
        header_row = ['Rank'] + [f'Model ID{i+1}' for i in range(num_models)] + ['All Classes Accuracy'] + ['All Classes Loss']
        writer.writerow(header_row)

        for rank in range(wsize):
            accuracy_row = [rank+1] + accuracy_matrix[rank] + [all_classes_accuracy[rank]] + [all_classes_loss[rank]]
            writer.writerow(accuracy_row)

        avg_row = ['Average'] + accuracy_avg.tolist() + [all_classes_accuracy_avg] + [all_classes_loss_avg]
        writer.writerow(avg_row)
def save_model(model, path):
    try:
        torch.save(model.state_dict(), path)
        print(f"Model saved successfully: {path}")
    except Exception as e:
        print(f"Error saving the model: {str(e)}")

def calculate_accuracy_matrix(merged_models, testset, trainloader, batch_size, wsize, datasetname, data_dist, device, testloader, save_path_model, phase, exp, opt, mode):
    accuracy_matrix = [[None] * wsize for _ in range(wsize)]
    all_classes_accuracy = []
    all_classes_loss = []

    for rank, model in enumerate(merged_models):
        model.to(device)

        model_path = os.path.join(save_path_model, f'model_{rank}.pth')
        save_model(model, model_path)

        for model_id in range(wsize):
            ctestloader = get_partition_dataloader(testset, data_dist, batch_size, wsize, datasetname, model_id, phase)
            
            # Calculate accuracy for both training and testing models
            accuracy, loss = test_accuracy(model, ctestloader, device)
            accuracy_matrix[rank][model_id] = accuracy

        accuracy, tloss = test_accuracy(model, testloader, device)
        all_classes_accuracy.append(accuracy)
        taccuracy, loss = test_accuracy(model, trainloader, device)
        all_classes_loss.append(loss)
        print("accuracy", all_classes_accuracy[rank])

        if mode == "merging" and exp == 1 and (opt == "DIMAT" or opt == "consensus_averaging"):
            for j in range(1, wsize):
                accuracy_matrix[j][:] = accuracy_matrix[0][:]
                # print("accuracy_matrix[j][:]", accuracy_matrix[j][:])
                all_classes_accuracy.append(accuracy)
                all_classes_loss.append(loss)
                model_path = os.path.join(save_path_model, f'model_{j}.pth')
                save_model(model, model_path)
                print("accuracy", all_classes_accuracy[j])
            break

    return accuracy_matrix, all_classes_accuracy, all_classes_loss        

if __name__ == "__main__":
    
    # Set the seed
    random_seed = args.seed
    seed_everything(random_seed)
    
    # Set device (GPU if available, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(device)
    # Unpack command-line arguments
    wsize = args.wsize
    experiment = args.exp
    width_multiplier = args.width_multiplier
    checkpoint_folder = args.ckp
    opt = args.opt
    save_folder = args.save_folder
    merg_itr = args.merg_itr
    data_dist = args.data_dist
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    training = args.training
    merg_itr_init = args.merg_itr_init
    optimizer = args.optim
    num_classes = args.num_classes
    datasetname = args.dataset
    match = args.match
    matrix = args.matrix 
    subdata = args.subdata
    numdata = args.numdata
    arch = args.model
    randominit = args.randominit
    imgntdir = args.imgntdir
    diffinit = args.diffinit
    phase = args.phase
    match_func = match_tensors_zipit
    
    # Define the metric_classes based on the --matrix argument
    if matrix == 'cov':
        metric_classes = (CovarianceMetric, MeanMetric)
    elif matrix == 'pycov':
        metric_classes = (Py_CovarianceMetric, MeanMetric)
    elif matrix == 'corr':
        metric_classes = (CorrelationMetric, MeanMetric)
    elif matrix == 'cossi':
        metric_classes = (CossimMetric, MeanMetric)
        
    if randominit == "True":
        init_path = 'randominit'
        if diffinit == "True":
            init_path = 'randomdiffinit'
        data_dist = args.data_dist
    else:
        init_path = 'trianedinit'


    opt_path = opt

    data_loader = Data_Loader(datasetname, batch_size, imgntdir)
    trainset, testset, trainloader, testloader, num_classes = data_loader.load_data()
            
    architectures = {
        'resnet50': (resnet50_graph, resnet50_permutation_spec),
        'resnet20': (resnet20_graph, resnet20_permutation_spec),
        'vgg16': (vgg16_graph, vgg16_permutation_spec)
    }

    if arch in architectures:
        graph_func, perm_spec = architectures[arch]
    else:
        raise ValueError(f"Architecture '{arch}' is not supported.")

 

    if arch != "vgg16" and opt == "WM":
        print('Weight Matching is only available for VGG architecture.')
        sys.exit() 
        
        
    
    # Split the path into components using the '/' delimiter
    path_components = checkpoint_folder.split('/')

    # Search for the component containing 'data_dist'
    for i, component in enumerate(path_components):
        if 'data_dist_' in component:
            # Extract the 'datadist' value
            data_dist = component.split('_')[2]
            print(data_dist)
            break
    merg_itr_in = merg_itr_init
    for merg_i in range(merg_itr_in, merg_itr):
        print("Iteration Number:", merg_i)
        # Determine training path
        training_path = f"{num_epochs}_epochs_per_i" if training else 'no_training'
        if merg_itr_init != 0:
            if training:
                checkpoint_folder = os.path.join(save_folder, 'Models', init_path, datasetname, arch, training_path,'data_dist_%s' % (data_dist), 
                                 'models_no%d' % wsize, '%s' % opt_path, 'graph_%d' % experiment, "trained", 'random_seed_%s' % (random_seed))
                print('load models from the last itration')
            else:
                checkpoint_folder = os.path.join(save_folder, 'Models', init_path, datasetname, arch, training_path,'data_dist_%s' % (data_dist), 
                                 'models_no%d' % wsize, '%s' % opt_path, 'graph_%d' % experiment, "merged", 'random_seed_%s' % (random_seed))
            models = load_models(arch, num_classes, datasetname, width_multiplier, device, wsize, checkpoint_folder)
            merg_itr_init = 0
        elif merg_i >= 1:
            models = merged_models
            print('new models loaded')
        else:
            models = load_models(arch, num_classes, datasetname, width_multiplier, device, wsize, checkpoint_folder, randominit, diffinit)
            print('Initializing the models.')
        
        

                
        
        # Create save path
        save_path = os.path.join(save_folder, 'Accuracy_Matrix', init_path, datasetname, arch, training_path,'data_dist_%s' % (data_dist), 
                                 'models_no%d' % wsize, '%s' % opt_path, 'graph_%d' % experiment, 'merg_itr_%d' % (merg_i + 1), 'random_seed_%s' % (random_seed))
        save_path_merged_model = os.path.join(save_folder, 'Models', init_path, datasetname, arch, training_path,'data_dist_%s' % (data_dist), 
                                 'models_no%d' % wsize, '%s' % opt_path, 'graph_%d' % experiment, "merged", 'random_seed_%s' % (random_seed))
        save_path_trained_model = os.path.join(save_folder, 'Models', init_path, datasetname, arch, training_path,'data_dist_%s' % (data_dist), 
                                 'models_no%d' % wsize, '%s' % opt_path, 'graph_%d' % experiment, "trained", 'random_seed_%s' % (random_seed))
        
        
        if not os.path.exists(save_path):
            os.makedirs(save_path)
            
        if not os.path.exists(save_path_merged_model):
            os.makedirs(save_path_merged_model)
        
        if not os.path.exists(save_path_trained_model):
            os.makedirs(save_path_trained_model)
            
        # Load connectivity data
        with open('Connectivity/%s_%s.json' % (wsize, experiment), 'r') as f:
            cdict = json.load(f)
        
        
        # Merge models
        merged_models = merge_models(models, cdict, opt, experiment, graph_func, device, num_classes, trainloader, testloader, match_func, metric_classes, perm_spec)

        # merged_models = models
        
        print("Calculating Accuracy Matrix")
        # Calculate accuracy matrix
        mode='Merging'
        accuracy_matrix, all_classes_accuracy, all_classes_loss = calculate_accuracy_matrix(merged_models, testset, trainloader, batch_size, wsize, 
                                                                                            datasetname, data_dist, device, testloader, save_path_merged_model, phase, experiment, opt, mode)

        # Write accuracy matrix to CSV
        output_file_testing = os.path.join(save_path, 'accuracy_matrix.csv')
        write_accuracy_matrix_to_csv(output_file_testing, accuracy_matrix, all_classes_accuracy, all_classes_loss)

        # Train the models in the training copy and write the accuracy matrix to CSV
        if training and ((merg_i < merg_itr - 1) or merg_i == 0):
            print("training")
            train_models(merged_models, training, wsize, num_epochs, batch_size, 
                         datasetname, data_dist, device, optimizer, trainset, phase)
            output_file_training = os.path.join(save_path, 'accuracy_matrix_training.csv')
            mode='training'
            accuracy_matrix, all_classes_accuracy, all_classes_loss = calculate_accuracy_matrix(merged_models,
                                                                                                  testset, trainloader, batch_size, 
                                                                                                 wsize,
                                                                                                 datasetname,
                                                                                                 data_dist,
                                                                                                 device, testloader, save_path_trained_model, phase, experiment, opt, mode)
            write_accuracy_matrix_to_csv(output_file_training, accuracy_matrix, all_classes_accuracy,
                                         all_classes_loss)
            

        
            
