import os
import utils
import models
import data
import json
from collaborative import Collab
from find_dom_set import *
import torch
import torch.distributed as dist
import torch.nn.functional as F
import argparse
import numpy as np
import time
from data import Data_Loader
from dataloader import get_partition_dataloader
import csv
import subprocess

def seed_everything(random_seed, myrank):
    random.seed(random_seed)
    os.environ['PYTHONHASHSEED'] = str(random_seed)
    np.random.seed(int(random_seed))
    torch.manual_seed(myrank)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Setting device to local rank (torch.distributed.launch)
    if args.use_cuda:
        devices = os.environ['CUDA_VISIBLE_DEVICES'].strip().split(',')
        #print(devices,flush=True)
        per_device_ranks = int(wsize/len(devices)) + 1
        print('Device assignment: %s , %s'%(args.local_rank, int(args.local_rank/per_device_ranks)),flush=True)
        torch.cuda.set_device(int(args.local_rank/per_device_ranks))

if __name__ == '__main__':
    # initialize MPI environment 
    # dist.init_process_group(backend="mpi")
    dist.init_process_group(backend="gloo", init_method='env://')
    rank = dist.get_rank()
    wsize = dist.get_world_size()  # number of processes = num_workers + 1  
    server_rank = -1 # this process is the server

    #############################################################################################################################
    #                                          setup code shared by workers and server
    #############################################################################################################################

    parser = argparse.ArgumentParser(description='PyTorch Example')
    parser.add_argument('--MCDS', action="store_true", default= False, help='If you need to compute MCDS for DSMA method')
    parser.add_argument('--save_folder', type=str, default='results', help='Folder path to save the models')
    parser.add_argument('--MCD_pi', action="store_true", help='If you need to compute MCDS for DSMA method')
    parser.add_argument('--use_cuda', action="store_false", help='Use CUDA if available')
    parser.add_argument('--data',type=str, default='cifar100', choices=("cifar10", "cifar100", "tinyimgnt", "Agdata", "Agdata-small"), help='Define the data used for training')
    parser.add_argument('--ckp', type=str, default=None, help='directory to checkpoints')
    parser.add_argument('--data_dist',type=str, default='non-iid', choices=("iid", "non-iid", "non-iid-PF", "non-iid-Ag-small"), help='Define the data distribution')
    parser.add_argument('--phase',type=str, default='finetune', choices=('pretrain', 'finetune'), help='Define the data distribution')
    parser.add_argument('--model',type=str,default='resnet20', help='Define the model used for training',
        choices=('resnet20', 'resnet50', 'vgg16', "LR","CNN","Big_CNN","FCN","stl10_CNN","mnist_CNN","supmnist",
                 "PreResNet110","WideResNet28x10", 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet1202',
                'VGG11', 'VGG13', 'VGG16', 'VGG19', 'aumnist', 'aucifar', 'aumnist2', 'aucifar_Res', 'VGG11_mnist'))
    parser.add_argument('--opt', type=str, default='SGP', help='Optimizer choices', 
                        choices=('sgd','CGA','adam','adagrad','adadelta','nesterov','CDSGD','CDMSGD','SGA','LGA','SGP','SwarmSGD', 'DSMA', 'CompLGA', 'LDSGD'))
    parser.add_argument('--batch_size', type=int, default=100, help='Define batch size for training')
    parser.add_argument('--epochs', type=int, default=200, help='Define num epochs for training')
    parser.add_argument('--exp', type=int, default=1, help='Experiment number of connectivity json')
    parser.add_argument("--randominit", type=str, default="False" , help='start models with random initialization')
    parser.add_argument("--diffinit", type=str, default="False" , help='start models with random and different initialization')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR', help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',help='Momentum (default: 0.9)')
    parser.add_argument('--I1',type=int,default=10, metavar='I1', help='I1 for LD-SGD')
    parser.add_argument('--I2',type=int,default=20,metavar='I2', help='I2 for LD-SGD')
    parser.add_argument('-v','--verbosity', type=int, default=1, help='verbosity of the code for debugging, 0==>No outputs, 1==>graph level outputs, 2==>agent level outputs')
    parser.add_argument('-log','--log_interval', type=int, default=1,help='How many epochs to wait before logging training results and models')
    parser.add_argument('--num_epochs', type=int, default=2, help='Number of epochs for training')
    parser.add_argument('--scheduler', action='store_true', default=True, help='Apply LR scheduler: step')
    parser.add_argument('-sche_step','--LR_sche_step', type=int, default=1, help='Stepsize for LR scheduler') # For StepLR
    parser.add_argument('-sche_lamb','--LR_sche_lamb', type=float, default=0.981, help='Lambda for LR scheduler') # For StepLR
    parser.add_argument('-meth', '--train_method', default = 'sup', type = str, help = 'method of trainig. E.g. sup unsup')
    # ================ For torch.distributed.launch multi-process argument ================
    parser.add_argument('--local-rank', type=int, help='Required argument for torch.distributed.launch, similar as rank')
    parser.add_argument('--seed', type=str, default='42',  help="Seed value for reproducibility")
    
    # ================ Not Implemented Yet ================
    parser.add_argument('-w','--omega', default=0.5, type=float, help='omega value of the generalized consensus')

    # ================================================

    args = parser.parse_args()

    random_seed = args.seed
    seed_everything(random_seed, str(int(random_seed)+rank))
    #torch.manual_seed(123)
    # batch_size=args.batch_size
    # num_epochs = args.epochs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    with open('Connectivity/%s_%s.json'%(wsize,args.exp), 'r') as f: #wsize = number of agents
        cdict = json.load(f)

    neighbors = [i[0] for i in enumerate(cdict['connectivity'][rank]) if i[1]>=0] # will include all rank, but required  
    # neighbors = [i[0] for i in enumerate(cdict['connectivity'][rank]) if i[1]>0] # only includes connected neigh, but doesn't work well with all_gather() with new_group()
    if args.randominit == "True":
        init_path = 'randominit'
        args.ckp = None
        if args.diffinit == "True":
            random_seed = args.seed
            seed_everything(random_seed, str(int(random_seed)+rank))
            init_path = 'randomdiffinit'
        else:
            random_seed = args.seed
            seed_everything(random_seed, random_seed)
        data_dist = args.data_dist
    else:
        init_path = 'trianedinit'

    # Create save folder
    if args.MCD_pi:
        folder_name = os.path.join("log",str(wsize)+'_agents', args.data_dist, args.data, args.model,str(cdict['graph_type']),args.opt+'_DOM_pi_'+str(args.momentum))
    else:
        #folder_name = os.path.join("log",str(wsize)+'_agents', args.data_dist, args.data, args.model,str(cdict['graph_type']),args.opt+'_'+str(args.momentum))
        # folder_name = os.path.join("log",str(wsize)+'_agents', args.data_dist, args.data, args.model,str(cdict['graph_type']),args.opt+'_'+str(args.momentum))
        folder_name = os.path.join(args.save_folder, 'Accuracy_Matrix', init_path, args.data, args.model, f"with_training_epo{args.num_epochs}" ,'data_dist_%s' % (args.data_dist), 
                                 'models_no%d' % wsize, '%s' % args.opt, 'graph_%d' % args.exp)
    if rank == 0:
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        if not os.path.exists('%s/visualizations'%(folder_name)):
            os.makedirs('%s/visualizations'%(folder_name))

    dom_set = []
    pi_dom = []
    connectivity = cdict['connectivity']
    if args.MCDS:
        time1 = time.process_time()
        dom_set = find_dom(connectivity, folder_name+'/visualizations', MCDS =True)
        print("############################################################3")
        print(f"{time.process_time() - time1} seconds takes to compute MCDS")
        print("############################################################3")
        connect_dom, pi_dom = gen_connect_dom(dom_set, connectivity)
        PI_MCD = find_PI(connectivity,pi_dom, dom_set)
        #print(PI_MCD)
        #print("dominating set connectivity matrix =", connect_dom)
        #print("dominating set pi matrix =", pi_dom)

    if not args.MCD_pi:
        pi_rank = cdict['pi'][rank]
    else:
        pi_rank = PI_MCD[rank]


    argdict = {
        'dist':dist,
        'MCD_pi': args.MCD_pi,
        'use_cuda':args.use_cuda,
        'data':args.data,
        'model_arch':args.model,
        'ckp':args.ckp,
        'experiment':args.exp,
        'graph_type':cdict['graph_type'],
        'data_dist':args.data_dist,
        'server_rank':server_rank,
        'epochs':args.epochs,
        'batch_size':args.batch_size,
        'optimizer':args.opt,
        'device':device,
        'wsize':wsize,
        'num_workers':wsize,
        'DS': dom_set,
        'random_seed': random_seed,
        'neighbors': neighbors,
        'pi_dom': pi_dom,
        'pi': pi_rank,
        'rank':rank,
        'lr':args.lr,
        'scheduler':args.scheduler,
        'LR_sche_step':args.LR_sche_step,
        'LR_sche_lamb':args.LR_sche_lamb,
        'momentum':args.momentum,
        'verbose':args.verbosity,
        'log_interval':args.log_interval,
        'log_folder':folder_name,
        'train_method':args.train_method,
        'I1':args.I1,
        'I2':args.I2,
        'seed': args.seed,
        'num_epochs':args.num_epochs,
        }

    print(f"Rank {rank}:",pi_rank)
    imgntdir = "tiny-imagenet-200"
    #############################################################################################################################
    #                                                 workers' setup code
    ############################################################################################################################# 
    data_loader = Data_Loader(args.data, args.batch_size, imgntdir)
    trainset, testset, dataloader, testdataloader, num_classes = data_loader.load_data()
    dataloader = get_partition_dataloader(trainset, args.data_dist, args.batch_size, wsize, args.data, rank, args.phase)
    #testdataloader = get_partition_dataloader(testset, args.data_dist, args.batch_size, wsize, args.data, rank, phase)
    data_dim = np.shape(testset[0][0].numpy())
    #print("data_dim",data_dim)
    #dataloader = data.LoadData(**argdict) ### data is divided between agents based on the local rank, so this dataloader is different for different ranks
    #testdataloader, data_dim = data.LoadTestData(**argdict) ### test data is not different for different ranks
    #print("data_dim =", data_dim)
    model, opt, criterion, LR_scheduler = models.LoadModel(data_dim, **argdict) # workers and servers all have a model! ## the model, optimizer, and loss are same for all of them.
    dist.barrier()
    trainer = Collab(dataloader, testdataloader, model, opt, criterion, LR_scheduler, **argdict)
    start_time = time.time()  # Record the start time
    print("here")
    trainer.train()
    end_time = time.time()  # Record the end time
    elapsed_time = end_time - start_time
    print(f"Elapsed Time: {elapsed_time} seconds")
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
    csv_filename = os.path.join(folder_name,'time_results.csv')
    csv_header = ['Rank', 'Elapsed Time (s)', 'GPU Memory Usage (GB)', 'opt', 'gpuinfo', 'Total Memory (GB)', 'Memory Reserved (GB)', 'Memory Allocated (GB)']
    csv_row = [rank, elapsed_time, gpu_memory_usage, opt, gpuinfo, total_memory_gb, memory_reserved_gb, memory_allocated_gb]


    with open(csv_filename, mode='a') as csv_file:
        csv_writer = csv.writer(csv_file)
        # Check if it's the first row to write the header
        if csv_file.tell() == 0:
            csv_writer.writerow(csv_header)
        csv_writer.writerow(csv_row)

#### Command to run the code for DSMA:

#  export CUDA_VISIBLE_DEVICES=0
# python -m torch.distributed.launch --nnodes 1 --nproc_per_node 5 main.py --experiment 1 --epochs 300 --opt CGA --use_cuda
# Results will be saved in the "log" folder