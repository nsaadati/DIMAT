import time, os, json, sys, csv
import torch
from Optimizers import *
from collections import defaultdict

class Trainer(object):
    """docstring for Trainer"""
    def __init__(self, dataloader, testdataloader, model, opt, criterion, LR_scheduler, **kwargs):
        self.dataloader = dataloader
        self.testdataloader = testdataloader
        self.model = model
        self.optimizer = opt
        self.criterion = criterion
        self.scheduler = LR_scheduler
        self.kwargs = kwargs
        self.epochs = kwargs['epochs']
        self.rank = kwargs['rank']
        self.wsize = kwargs.get('wsize', -1)
        self.dist = kwargs['dist']
        self.device = kwargs['device']
        self.log_interval = kwargs['log_interval']
        self.graph_type = kwargs['graph_type']
        assert self.device is not None

        ''' Initialize log placeholders
         # Global loss, acc and training time are stored in Worker 0
         '''
        self.worker_train_loss_hist = []      # Worker training loss during(/within) epoch update (with param. update between minibatch)
        
        if self.rank == 0:
            self.global_train_loss_hist = []  # Graph level traininig loss (with param. update between minibatch)

        self.worker_trainloader_loss_hist = []  # Worker trainloader loss at every end of epoch
        self.worker_testloader_loss_hist = []   # Worker testloader loss at every end of epoch
        self.worker_trainloader_acc_hist = []   # Worker trainloader accuracy at every end of epoch
        self.worker_testloader_acc_hist = []    # Worker testloader accuracy at every end of epoch

        if self.rank == 0:
            self.global_trainloader_loss_hist = []  # Global/Average trainloader loss at every end of epoch
            self.global_testloader_loss_hist = []   # Global/Average testloader loss at every end of epoch
            self.global_trainloader_acc_hist = []   # Global/Average trainloader accuracy at every end of epoch
            self.global_testloader_acc_hist = []    # Global/Average testloader accuracy at every end of epoch

        self.worker_train_time_hist = []          # Each worker's epoch training time
        
        if self.rank == 0:
            self.global_avg_train_time_hist = []  # Average worker's epoch training time


    def hand_shake(self):
        raise NotImplementedError('must have a hand_shake defined')

    def train_epoch(self, epoch_id):
        raise NotImplementedError('must have a train_epoch defined')

    def log(self, epoch_id):
        folder_name = self.kwargs['log_folder']
        num_epochs = self.kwargs['num_epochs']
        random_seed = self.kwargs['seed']
        if self.rank == 0:
            print('Epoch %s finished with global training loss of %s and test accuracy of %s in %0.2f secs'%(epoch_id+1, self.global_train_loss_hist[-1], self.global_testloader_acc_hist[-1], self.global_avg_train_time_hist[-1]))
            if (epoch_id + 1) % num_epochs == 0:
                all_classes_accuracy = []
                all_classes_loss = []
                merg_i = (epoch_id + 1) // num_epochs
                print(merg_i)
                save_path = os.path.join(folder_name, 'merg_itr_%d' % (merg_i), 'random_seed_%s' % (random_seed))
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                all_classes_accuracy.append(self.global_testloader_acc_hist[-1])
                all_classes_loss.append(self.global_train_loss_hist[-1])
                output_file = os.path.join(save_path, 'accuracy_matrix.csv')
                with open(output_file, 'w', newline='') as file:
                    writer = csv.writer(file)
                    header_row = ['All Classes Accuracy'] + ['All Classes Loss']
                    writer.writerow(header_row)
                    accuracy_row = [all_classes_accuracy[0]] + [all_classes_loss[0]]
                    writer.writerow(accuracy_row)
        # worker code 
        if (self.rank != self.server_rank):
            if ((epoch_id+1)%self.log_interval == 0) or ((epoch_id+1)==self.epochs): # epoch_id now based on 1-indexing
                # Saving workers result
                log = {}
                log['optimizer'] = self.kwargs['optimizer']
                log['worker_id'] = self.rank
                log['batch_size'] = self.kwargs['batch_size']
                log['nb_epoch'] = epoch_id+1 # changing to 1-indexed
                log['data'] = self.kwargs['data']
                log['model'] = self.kwargs['model_arch']
                log['momentum'] = self.kwargs['momentum']
                log['train_loss'] = self.worker_train_loss_hist                 # Worker training loss during(/within) epoch update (with param. update between minibatch)
                log['train_epoch_time'] = self.worker_train_time_hist           # worker epoch training time
                log['trainloader_loss'] = self.worker_trainloader_loss_hist     # worker trainloader loss at the end of every epoch
                log['trainload_acc'] = self.worker_trainloader_acc_hist         # worker trainloader acc at the end of every epoch
                log['testloader_loss'] = self.worker_testloader_loss_hist       # worker testloader loss at the end of every epoch
                log['testloader_acc'] = self.worker_testloader_acc_hist         # worker testloader acc at the end of every 

                log_fn = os.path.join(folder_name,'rank_'+str(self.rank)+'.json')

                with open(log_fn, 'w') as fp1:
                    json.dump(log, fp1, indent=4, sort_keys=False)
                if self.verbose >= 2:
                    print('Worker %s finished epoch %s with a trainloader loss of %0.2f.'%(self.rank, epoch_id+1, self.worker_trainloader_loss_hist[-1]))
                # Saving global result
                if self.rank == 0:
                    log = {}
                    log['optimizer'] = self.kwargs['optimizer']
                    log['batch_size'] = self.kwargs['batch_size']
                    log['nb_epoch'] = epoch_id+1 # changing to 1-indexed
                    log['data'] = self.kwargs['data']
                    log['model'] = self.kwargs['model_arch']
                    log['momentum'] = self.kwargs['momentum']
                    log['num_agents'] = self.kwargs['wsize']
                    log['train_loss'] = self.global_train_loss_hist                 # Global training loss during(/within) epoch update (with param. update between minibatch)
                    log['train_epoch_time'] = self.global_avg_train_time_hist       # Global epoch training time
                    log['trainloader_loss'] = self.global_trainloader_loss_hist     # Global trainloader loss at the end of every epoch
                    log['trainload_acc'] = self.global_trainloader_acc_hist         # Global trainloader acc at the end of every epoch
                    log['testloader_loss'] = self.global_testloader_loss_hist       # Global testloader loss at the end of every epoch
                    log['testloader_acc'] = self.global_testloader_acc_hist         # Global testloader acc at the end of every epoch

                    log_fn = os.path.join(folder_name,'global.json')

                    with open(log_fn, 'w') as fp1:
                        json.dump(log, fp1, indent=4, sort_keys=False)

                checkpoint_path = os.path.join(folder_name,'model_'+str(self.rank)+'.tar')
                if epoch_id % 20 == 0 or epoch_id == 1:
                    torch.save({'epoch': epoch_id,
                                'model_state_dict': self.model.state_dict(),
                                'optimizer_state_dict': self.optimizer.state_dict(),
                                'loss': self.worker_trainloader_loss_hist}, checkpoint_path)
                
    def train(self):
        # training starts
        if self.rank is self.workers[0]:
            print('Training begun')
        TotalStrtTime = time.monotonic()

        for epoch_id in range(self.epochs):
            EpochStrtTime = time.monotonic()
            # if self.kwargs['optimizer'] == 'SwarmSGD':
            self.train_epoch(epoch_id)
            self.dist.barrier()  # synchronize all workers after each epochs
            self.exec_time = time.monotonic() - EpochStrtTime
            self.worker_train_time_hist.append(self.exec_time)
            self.test_epoch(epoch_id)
            self.compute_globals()
            self.log(epoch_id)
            self.dist.barrier()  # synchronize all workers after each epochs

        # now finishes all epochs, do an all_reduce between workers 
        # self.dist.barrier(self.distgroup)  # synchronize all workers 
    

        if self.rank == 0:
            TotalTime = (time.monotonic()-TotalStrtTime)
            print("Execution of training in {}".format(TotalTime))



class SingleTrainer(object):
    """docstring for SingleTrainer"""
    def __init__(self, dataloader, testdataloader, model, opt, criterion, LR_scheduler, **kwargs):
        super(SingleTrainer, self).__init__(dataloader, testdataloader, model, opt, criterion, LR_scheduler, **kwargs)
        assert self.rank == 0
        assert self.wsize == 1
        self.dataloader = dataloader
        self.testdataloader = testdataloader
        self.model = model
        self.optimizer = opt
        self.criterion = criterion
        self.scheduler = LR_scheduler
        self.kwargs = kwargs
        self.epochs = kwargs['epochs']
        self.device = kwargs['device']
        self.log_interval = kwargs['log_interval']
        assert self.device is not None

        ''' Initialize log placeholders
         # Global loss, acc and training time are stored in Worker 0
         '''
        self.worker_train_loss_hist = []      # Worker training loss during(/within) epoch update (with param. update between minibatch)
        self.global_train_loss_hist = []  # Graph level traininig loss (with param. update between minibatch)
        self.worker_trainloader_loss_hist = []  # Worker trainloader loss at every end of epoch
        self.worker_testloader_loss_hist = []   # Worker testloader loss at every end of epoch
        self.worker_trainloader_acc_hist = []   # Worker trainloader accuracy at every end of epoch
        self.worker_testloader_acc_hist = []    # Worker testloader accuracy at every end of epoch

        self.global_trainloader_loss_hist = []  # Global/Average trainloader loss at every end of epoch
        self.global_testloader_loss_hist = []   # Global/Average testloader loss at every end of epoch
        self.global_trainloader_acc_hist = []   # Global/Average trainloader accuracy at every end of epoch
        self.global_testloader_acc_hist = []    # Global/Average testloader accuracy at every end of epoch

        self.worker_train_time_hist = []          # Each worker's epoch training time

        self.global_avg_train_time_hist = []  # Average worker's epoch training time

    def hand_shake(self):
        pass # No need for hand shake for a single agent

    def train_epoch(self):
       pass 
        
