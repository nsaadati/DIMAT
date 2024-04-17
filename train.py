import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from torch.autograd import Variable
import time
import os
from copy import deepcopy
import numpy as np
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim
from torchvision.transforms import ToTensor
import random
import argparse
import csv
from Models.resnet import ResNet
from Models.resnetzip import resnet20
import torch.utils.data as data
from torch.utils.data import Dataset, SubsetRandomSampler
from Models.vgg import vgg16
from dataloader import get_partition_dataloader
from data import Data_Loader
from LoadModel import load_model, load_models


def seed_everything(random_seed):
    random.seed(random_seed)
    os.environ['PYTHONHASHSEED'] = str(random_seed)
    np.random.seed(int(random_seed))
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
def train_model(model, epochs, train_loader, device, optimizer_type, softmax=False):
    model.train()

    if optimizer_type == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=0.001)
    elif optimizer_type == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    else:
        raise ValueError("Unsupported optimizer type")

    for epoch in range(epochs):  # Adjust the number of epochs as needed
        running_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            if softmax:
                output = F.log_softmax(output, dim=1)
            loss = F.nll_loss(output, target)
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if batch_idx % 10 == 9:    # Print the loss every 10 mini-batches
                # print(f'Epoch: {epoch + 1}, Batch: {batch_idx + 1}, Loss: {running_loss / 10:.3f}')
                running_loss = 0.0
    print('Training complete')

    train_accuracy, tloss = test_accuracy(model, train_loader, device)
    print('Training Accuracy', train_accuracy)
    return train_accuracy, tloss

def test_accuracy(model, dataloader, device, softmax=True):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            if softmax:
                output = F.log_softmax(output, dim=1)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()


    test_loss /= len(dataloader.dataset)
    acc = 100. * correct / len(dataloader.dataset)

    return acc, test_loss  # Return both accuracy and average loss    

if __name__ == "__main__":
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Train multiple models on CIFAR-100')
    parser.add_argument('--phase',type=str, default='pretrain', choices=('pretrain', 'finetune'), help='Define the data distribution')
    parser.add_argument('--num_models', type=int, default=5, help='Number of models to train')
    parser.add_argument('--data_dist', type=str, default='non-iid', choices=['iid', 'non-iid', 'non-iid-PF'], help='Degree of nonidness')
    parser.add_argument('--num_classes', type=int, default=100, help='Number of all the classes')
    parser.add_argument('--random_seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--init_seed', type=bool, default=False, help='use initialization seed for training')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs for training')
    parser.add_argument('--depth', type=int, default=22)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--opt', type=str, default='adam')
    parser.add_argument('--model', type=str, default='resnet20', choices=['resnet20', 'resnet50', 'vgg16'],
                        help="Specify the architecture for models. Default is 'resnet20'.")
    parser.add_argument('--dataset', type=str, default='cifar100', choices=['cifar100', 'tinyimgnt', 'cifar10', 'mnist'])
    parser.add_argument('--imgntdir', type=str, default='tiny-imagenet-200', help="directory for tiny imagenet dataset")
    parser.add_argument('--width_multiplier', type=int, default=8)
    parser.add_argument('--save_folder', type=str, default='checkpoint', help='Folder path to save the models')
    parser.add_argument('--seed', type=str, default='42',  help="Seed value for reproducibility")
    args = parser.parse_args()

    
    # Set the seed
    random_seed = args.seed
    seed_everything(random_seed)
    
    # Set the random seed for reproducibility
    random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    num_models = args.num_models
    num_classes = args.num_classes
    init_seed_flag = args.init_seed
    epochs = args.epochs
    depth = args.depth
    batch_size = args.batch_size
    optimizer = args.opt
    width_multiplier = args.width_multiplier
    save_folder = args.save_folder
    arch = args.model
    datasetname = args.dataset
    imgntdir = args.imgntdir
    phase = args.phase
    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(device)
    
    
    data_loader = Data_Loader(datasetname, batch_size, imgntdir)
    trainset, testset, trainloader, testloader, num_classes = data_loader.load_data()
    
    # Create the save folder if it doesn't exist
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    if args.data_dist is not None:
        data_dists = [args.data_dist]
    else:
        data_dists = ['iid', 'non-iid']  
    num_trained_classes = num_classes
    for data_dist in data_dists:
        if data_dist == 'iid':
            num_trained_classes = num_classes
        elif data_dist == 'non-iid':
            num_trained_classes = num_classes // num_models
        model_accuracies = []
        #trainloaders, random_classes, num_trained_classes = get_trainloaders(num_models, trainset, num_classes, data_dist, batch_size)
        
        # Loop through each model and train them on different random classes
        for i in range(num_models):
            # Set the random seed for initialization
            if init_seed_flag:
                diffinit = "False"
                seed = random_seed
            else:
                initialization = "diff_initialization"
                seed = i + int(random_seed)
                diffinit = "True"
            
            model = load_model(arch, num_classes, datasetname, width_multiplier, device, seed, diffinit, path = None)
            agent_train_loader = get_partition_dataloader(trainset, data_dist, batch_size, num_models, datasetname, i, phase)
            # Train the model
            train_accuracy, tloss = train_model(model, epochs, train_loader = agent_train_loader, device=device, optimizer_type=optimizer, softmax=True)
            
                # Create a new DataLoader for the filtered dataset
            ctestloader = get_partition_dataloader(testset, data_dist, batch_size, num_models, datasetname, i, phase)

            ctest_accuracy, closs = test_accuracy(model, ctestloader, device)

            # Test accuracy on all classes
            atest_accuracy, atloss = test_accuracy(model, testloader, device)
            atrain_accuracy, aloss = test_accuracy(model, trainloader, device)
            # Save the trained model
            save_path = os.path.join(save_folder, datasetname, arch, initialization, 'models_no%d' % num_models, 'data_dist_%s' % (data_dist), 'num_epochs_%d' % (epochs), 'random_seed_%s' % (random_seed))
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            model_path = os.path.join(save_path, f'model_{i}.pth')
            torch.save(model.state_dict(), model_path)

                # Save accuracy information and trained classes to CSV
            model_info = {
                'Model': f'model_{i}',
                'Training Accuracy': train_accuracy,
                'Training Loss': tloss,
                'Trained Classes Test Accuracy': ctest_accuracy,
                'Trained Classes Test Loss': closs,
                'All Classes Test Accuracy': atest_accuracy,
                'All Classes Train Loss': aloss,
                'Number of Trained Classes': num_trained_classes,
                'Degree of Nonidness': data_dist
            }
            model_accuracies.append(model_info)

        # Save accuracy information and trained classes to a CSV file
        csv_path = os.path.join(save_path, 'model_accuracies.csv')
        fieldnames = ['Model', 'Training Accuracy', 'Training Loss', 'Trained Classes Test Accuracy', 'Trained Classes Test Loss', 'All Classes Test Accuracy', 'All Classes Train Loss', 'Number of Trained Classes', 'Degree of Nonidness']
        with open(csv_path, 'w', newline='') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(model_accuracies)

        print('All models trained and saved')
        print(f'Accuracy information saved to {csv_path}')