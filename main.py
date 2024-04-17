import argparse
import json
from Models.resnet import ResNet
import torch
from utils.weight_matching import find_permutation, apply_permutation, resnet20_permutation_spec, resnet50_permutation_spec, vgg16_permutation_spec
from tqdm import tqdm
import copy
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as T
import torch.nn.functional as F
import os
import csv
import ast
import os
import pdb
import clip
import random
import time
import numpy as np
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from itertools import product
import argparse
import pandas as pd
import pickle
from Models.resnetzip import resnet20
from train import train_model, test_accuracy
from utils.am_utils import *
from utils.model_merger import ModelMerge
from graphs.resnet_graph import resnet20 as resnet20_graph
from graphs.resnet_graph import resnet50 as resnet50_graph
from graphs.vgg_graph import vgg16 as vgg16_graph
from utils.matching_functions import match_tensors_zipit, match_tensors_optimal, match_tensors_permute, match_tensors_kmeans, match_tensors_randperm
from utils.metric_calculators import CovarianceMetric, MeanMetric, Py_CovarianceMetric, CorrelationMetric, CossimMetric
from torch.utils.data import SubsetRandomSampler
from torch.utils.data import Subset
from Models.vgg import vgg16
from dataloader import get_partition_dataloader
from data import Data_Loader
from LoadModel import load_model, load_models

# Create argument parser
parser = argparse.ArgumentParser(description='Load models and calculate merged model')
parser.add_argument('--num_models', type=int, default=5, help='Number of agents')
parser.add_argument('--exp', type=int, default=1, help='graph indicator')
parser.add_argument('--data_dist', type=str, default='non-iid', choices=['iid', 'non-iid', 'non-iid-PF'], help='Degree of nonidness')
parser.add_argument('--epochs', type=int, default=2, help='Number of epochs for training')
parser.add_argument('--merg_itr', type=int, default=1, help='the iteration number for merging')
parser.add_argument('--merg_itr_init', type=int, default=0, help='the iteration number that strats with')
parser.add_argument('--ckp', type=str, help='Path to the models checkpoint')
parser.add_argument('--save_folder', type=str, default='results', help='Folder path to save the models')
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument("--training", action="store_true" , help='train after merging or not')
parser.add_argument("--randominit", type=str, default="False" , help='start models with random initialization')
parser.add_argument("--diffinit", type=str, default="False" , help='start models with random and different initialization')
parser.add_argument('--phase',type=str, default='finetune', choices=('pretrain', 'finetune'), help='Define the data distribution')
parser.add_argument('--optim', type=str, default='adam')
parser.add_argument('--dataset', type=str, default='cifar100', choices=['cifar100', 'tinyimgnt', 'cifar10', 'mnist'])
parser.add_argument('--imgntdir', type=str, default='tiny-imagenet-200', help="directory for tiny imagenet dataset")
parser.add_argument('--opt', type=str, default='DIMAT', help='the algorithm to merge models',
                   choices= ('WM', 'WA', 'DIMAT'))
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
    model_state_dicts = [agent.state_dict() for agent in models]
    for m_idx, agent in enumerate(models):
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
    # model_state_dicts = [agent.state_dict() for agent in models]
    for m_idx, agent in enumerate(models):
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

def DIMAT(models, cdict, experiment, graph_func, device, num_classes, trainloader, testloader, covsave_path, corrsave_path):

    exp = experiment
    merged_models = []  
    match_func = match_tensors_permute
    metric_classes = (CovarianceMetric, MeanMetric)
    for m_idx, agent in enumerate(models):
        interp_w = []
        print('m_idx',m_idx)
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
            print('neighbor_id',neighbor_id)

            models[neighbor_id] = reset_bn_stats(models[neighbor_id], trainloader)
            temp_models.append(copy.deepcopy(models[neighbor_id]))
            interp_w.append(pi_m_idx[neighbor_id])
        covsave_path_mn = os.path.join(covsave_path, 'agent_id_%d' % m_idx, 'neighbor_id_%d' % len(models))
        corrsave_path_mn = os.path.join(corrsave_path, 'agent_id_%d' % m_idx, 'neighbor_id_%d' % len(models))
        #temp_model = copy.deepcopy(models[m_idx]).to(device)
        graphs = [ graph_func(agent).graphify() for agent in temp_models]
        del temp_models
        Merge = ModelMerge(*graphs, device=device)
        Merge.transform(
            merged_models[m_idx], 
            trainloader, 
            covsave_path = covsave_path_mn,
            corrsave_path = corrsave_path_mn,
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

def merge_models(models, cdict, opt, experiment, graph_func, device, num_classes, trainloader, testloader, covsave_path, corrsave_path, perm_spec):
    if opt == 'WM':
        merged_models = WM(models, cdict, perm_spec, device)
    elif opt == 'WA':
        merged_models = WA(models, cdict)
    elif opt == 'DIMAT':
        merged_models = DIMAT(models, cdict, experiment, graph_func, device, num_classes, trainloader, testloader, covsave_path, corrsave_path)
    return merged_models
def train_models(merged_models, training, num_models, epochs, batch_size, datasetname, data_dist, device, optimizer, trainset, phase):
    if not training:
        return                        
    for model_id, agent in enumerate(merged_models):
        model_trainloader = get_partition_dataloader(trainset, data_dist, batch_size, num_models, datasetname, model_id, phase)
        train_model(agent, epochs, train_loader=model_trainloader, device=device, optimizer_type=optimizer, softmax=True)
        

def write_accuracy_matrix_to_csv(output_file, accuracy_matrix, all_classes_accuracy, all_classes_loss):
    num_models = len(accuracy_matrix[0])
    accuracy_avg = np.mean(accuracy_matrix, axis=0)
    all_classes_accuracy_avg = np.mean(all_classes_accuracy)
    all_classes_loss_avg = np.mean(all_classes_loss)

    with open(output_file, 'w', newline='') as file:
        writer = csv.writer(file)
        header_row = ['Rank'] + [f'Model ID{i+1}' for i in range(num_models)] + ['All Classes Accuracy'] + ['All Classes Loss']
        writer.writerow(header_row)

        for rank in range(num_models):
            accuracy_row = [rank+1] + accuracy_matrix[rank] + [all_classes_accuracy[rank]] + [all_classes_loss[rank]]
            writer.writerow(accuracy_row)

        avg_row = ['Average'] + accuracy_avg.tolist() + [all_classes_accuracy_avg] + [all_classes_loss_avg]
        writer.writerow(avg_row)
def save_model(agent, path):
    try:
        torch.save(agent.state_dict(), path)
        print(f"Model saved successfully: {path}")
    except Exception as e:
        print(f"Error saving the model: {str(e)}")

def calculate_accuracy_matrix(merged_models, testset, trainloader, batch_size, num_models, datasetname, data_dist, device, testloader, save_path_model, phase, exp, opt, mode):
    accuracy_matrix = [[None] * num_models for _ in range(num_models)]
    all_classes_accuracy = []
    all_classes_loss = []

    for rank, agent in enumerate(merged_models):
        agent.to(device)

        model_path = os.path.join(save_path_model, f'model_{rank}.pth')
        save_model(agent, model_path)

        for model_id in range(num_models):
            ctestloader = get_partition_dataloader(testset, data_dist, batch_size, num_models, datasetname, model_id, phase)
            
            # Calculate accuracy for both training and testing models
            accuracy, loss = test_accuracy(agent, ctestloader, device)
            accuracy_matrix[rank][model_id] = accuracy

        # print("accuracy_matrix[rank][:]", accuracy_matrix[rank][:])
            
        # Calculate accuracy for all classes
        accuracy, tloss = test_accuracy(agent, testloader, device)
        all_classes_accuracy.append(accuracy)
        taccuracy, loss = test_accuracy(agent, trainloader, device)
        all_classes_loss.append(loss)
        print("accuracy", all_classes_accuracy[rank])

        if mode == "merging" and exp == 1 and (opt == "DIMAT" or opt == "WA"):
            for j in range(1, num_models):
                accuracy_matrix[j][:] = accuracy_matrix[0][:]
                # print("accuracy_matrix[j][:]", accuracy_matrix[j][:])
                all_classes_accuracy.append(accuracy)
                all_classes_loss.append(loss)
                model_path = os.path.join(save_path_model, f'model_{j}.pth')
                save_model(agent, model_path)
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
    num_models = args.num_models
    experiment = args.exp
    #depth = args.depth
    checkpoint_folder = args.ckp
    opt = args.opt
    save_folder = args.save_folder
    merg_itr = args.merg_itr
    #csv_file = args.csv
    data_dist = args.data_dist
    epochs = args.epochs
    batch_size = args.batch_size
    training = args.training
    merg_itr_init = args.merg_itr_init
    optimizer = args.optim
    datasetname = args.dataset
    model = args.model
    randominit = args.randominit
    imgntdir = args.imgntdir
    diffinit = args.diffinit
    phase = args.phase
        
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
            
    if model == 'resnet50':
        graph_func = resnet50_graph
        perm_spec = resnet50_permutation_spec
    elif model == 'resnet20':
        graph_func = resnet20_graph
        perm_spec = resnet20_permutation_spec
    elif model == 'vgg16':
        graph_func = vgg16_graph
        perm_spec = vgg16_permutation_spec

    subtrainloader = trainloader
        
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
    start_time = time.time()  # Record the start time
    for merg_i in range(merg_itr_in, merg_itr):
        print("merg_i", merg_i)
        print("merg_itr", merg_itr)
        # Determine training path
        training_path = f"with_training_epo{epochs}" if training else 'no_training'
        if merg_itr_init != 0:
            if training:
                checkpoint_folder = os.path.join(save_folder, 'Models', init_path, datasetname, model, training_path,'data_dist_%s' % (data_dist), 
                                 'models_no%d' % num_models, '%s' % opt_path, 'graph_%d' % experiment, "trained", 'random_seed_%s' % (random_seed))
                print('load from the last itration')
            else:
                checkpoint_folder = os.path.join(save_folder, 'Models', init_path, datasetname, model, training_path,'data_dist_%s' % (data_dist), 
                                 'models_no%d' % num_models, '%s' % opt_path, 'graph_%d' % experiment, "merged", 'random_seed_%s' % (random_seed))
            models = load_models(model, num_classes, datasetname, device, num_models, checkpoint_folder)
            merg_itr_init = 0
        elif merg_i >= 1:
            models = merged_models
            print('new models loaded')
        else:
            models = load_models(model, num_classes, datasetname, device, num_models, checkpoint_folder, randominit, diffinit)
            print('Initializing the models.')
        
        

                
        
        # Create save path
        save_path = os.path.join(save_folder, 'Accuracy_Matrix', init_path, datasetname, model, training_path,'data_dist_%s' % (data_dist), 
                                 'models_no%d' % num_models, '%s' % opt_path, 'graph_%d' % experiment, 'merg_itr_%d' % (merg_i + 1), 'random_seed_%s' % (random_seed))
        save_path_time = os.path.join(save_folder, 'timing', init_path, datasetname, model, training_path,'data_dist_%s' % (data_dist), 
                                 'models_no%d' % num_models, '%s' % opt_path, 'graph_%d' % experiment, 'merg_itr_%d' % (merg_itr), 'random_seed_%s' % (random_seed))
        save_path_merged_model = os.path.join(save_folder, 'Models', init_path, datasetname, model, training_path,'data_dist_%s' % (data_dist), 
                                 'models_no%d' % num_models, '%s' % opt_path, 'graph_%d' % experiment, "merged", 'random_seed_%s' % (random_seed))
        save_path_trained_model = os.path.join(save_folder, 'Models', init_path, datasetname, model, training_path,'data_dist_%s' % (data_dist), 
                                 'models_no%d' % num_models, '%s' % opt_path, 'graph_%d' % experiment, "trained", 'random_seed_%s' % (random_seed))
        covsave_path = os.path.join(save_folder, "covmatric", init_path, datasetname, model, training_path,'data_dist_%s' % (data_dist), 
                                    'models_no%d' % num_models, '%s' % opt_path, 'graph_%d' % experiment, 'merg_itr_%d' % (merg_i + 1), 'random_seed_%s' % (random_seed))
        corrsave_path = os.path.join(save_folder, "corrmatric", init_path, datasetname, model, training_path, 'data_dist_%s' % (data_dist),
                                     'models_no%d' % num_models,'%s' % opt_path, 'graph_%d' % experiment, 'merg_itr_%d' % (merg_i + 1), 'random_seed_%s' % (random_seed))
        
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if not os.path.exists(save_path_time):
            os.makedirs(save_path_time)
            
        if not os.path.exists(save_path_merged_model):
            os.makedirs(save_path_merged_model)
        
        if not os.path.exists(save_path_trained_model):
            os.makedirs(save_path_trained_model)
            
        # Load connectivity data
        with open('Connectivity/%s_%s.json' % (num_models, experiment), 'r') as f:
            cdict = json.load(f)
        
        
        # Merge models
        merged_models = merge_models(models, cdict, opt, experiment, graph_func, device, num_classes, trainloader, testloader, covsave_path, corrsave_path, perm_spec)

        # merged_models = models
        
        print("Calculating accuracy matrix")
        # Calculate accuracy matrix
        mode='merging'
        accuracy_matrix, all_classes_accuracy, all_classes_loss = calculate_accuracy_matrix(merged_models, testset, trainloader, batch_size, num_models, 
                                                                                            datasetname, data_dist, device, testloader, save_path_merged_model, phase, experiment, opt, mode)

        # Write accuracy matrix to CSV
        output_file_testing = os.path.join(save_path, 'accuracy_matrix.csv')
        write_accuracy_matrix_to_csv(output_file_testing, accuracy_matrix, all_classes_accuracy, all_classes_loss)

        # Train the models in the training copy and write the accuracy matrix to CSV
        if training and ((merg_i < merg_itr - 1) or merg_i == 0):
            print("training")
            train_models(merged_models, training, num_models, epochs, batch_size, 
                         datasetname, data_dist, device, optimizer, trainset, phase)
            output_file_training = os.path.join(save_path, 'accuracy_matrix_training.csv')
            mode='training'
            accuracy_matrix, all_classes_accuracy, all_classes_loss = calculate_accuracy_matrix(merged_models,
                                                                                                  testset, trainloader, batch_size, 
                                                                                                 num_models,
                                                                                                 datasetname,
                                                                                                 data_dist,
                                                                                                 device, testloader, save_path_trained_model, phase, experiment, opt, mode)
            write_accuracy_matrix_to_csv(output_file_training, accuracy_matrix, all_classes_accuracy,
                                         all_classes_loss)
    end_time = time.time()  # Record the end time
    elapsed_time = end_time - start_time
    print(f"Elapsed Time: {elapsed_time} seconds")
    gpu_memory_usage = torch.cuda.memory_allocated() / 1024**3  # Convert bytes to gigabytes
    print(f"GPU Memory Usage: {gpu_memory_usage:.2f} GB")
    gpu_properties = torch.cuda.get_device_properties(0)
    total_memory = gpu_properties.total_memory
    memory_reserved = torch.cuda.memory_reserved(0)
    memory_allocated = torch.cuda.memory_allocated(0)
    gpuinfo = torch.cuda.mem_get_info()

    # Calculate GPU memory usage in GB
    total_memory_gb = total_memory / (1024 ** 3)  # Convert bytes to gigabytes
    memory_reserved_gb = memory_reserved / (1024 ** 3)  # Convert bytes to gigabytes
    memory_allocated_gb = memory_allocated / (1024 ** 3)  # Convert bytes to gigabytes
    gpu_memory_usage = torch.cuda.memory_allocated() / 1024**3  # Convert bytes to gigabytes
    print("GPU Information:")
    print("Total Memory: {:.2f} GB".format(total_memory_gb))
    print("Memory Reserved: {:.2f} GB".format(memory_reserved_gb))
    print("Memory Allocated: {:.2f} GB".format(memory_allocated_gb))
    print("GPU Memory Usage: {:.2f} GB".format(gpu_memory_usage))

    # Save elapsed time to a CSV file
    for rank in range(0,num_models):
        csv_timing = os.path.join(save_path_time, 'time_results.csv')
        csv_header = ['Rank', 'Elapsed Time (s)', 'GPU Memory Usage (GB)', 'opt', 'gpuinfo', 'Total Memory (GB)', 'Memory Reserved (GB)', 'Memory Allocated (GB)']
        csv_row = [rank, elapsed_time, gpu_memory_usage, opt, gpuinfo, total_memory_gb, memory_reserved_gb, memory_allocated_gb]

        with open(csv_timing, mode='a') as csv_file:
            csv_writer = csv.writer(csv_file)
            # Check if it's the first row to write the header
            if csv_file.tell() == 0:
                csv_writer.writerow(csv_header)
            csv_writer.writerow(csv_row)

            

        
            
