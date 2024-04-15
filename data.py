import os
import torch
import torchvision
import torchvision.transforms as T
import numpy as np

class Data_Loader:
    def __init__(self, datasetname, batchsize, datasetdir=None):
        self.datasetname = datasetname
        self.batchsize = batchsize
        self.datasetdir = datasetdir

    def load_data(self):
        if self.datasetname == 'cifar100' or self.datasetname == 'CIFAR100':
            print("cifar100")
            return self.load_cifar100()
        elif self.datasetname == 'cifar10' or self.datasetname == 'CIFAR10':
            print("cifar10")
            return self.load_cifar10()
        elif self.datasetname == 'mnist' or self.datasetname == 'MNIST':
            print("mnist")
            return self.load_mnist()
        elif self.datasetname == 'tinyimgnt':
            print("tiny imagenet")
            return self.load_tinyimagenet()
        else:
            raise ValueError("Unsupported dataset: " + self.datasetname)

    def load_cifar100(self):
        CIFAR_MEAN = [125.307, 122.961, 113.8575]
        CIFAR_STD = [51.5865, 50.847, 51.255]
        normalize = T.Normalize(np.array(CIFAR_MEAN) / 255, np.array(CIFAR_STD) / 255)
        denormalize = T.Normalize(-np.array(CIFAR_MEAN) / np.array(CIFAR_STD), 255 / np.array(CIFAR_STD))
        train_transform = T.Compose([T.RandomHorizontalFlip(), T.RandomCrop(32, padding=4), T.ToTensor(), normalize])
        test_transform = T.Compose([T.ToTensor(), normalize])
        trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=train_transform)
        testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=test_transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=self.batchsize, shuffle=True, num_workers=4)
        testloader = torch.utils.data.DataLoader(testset, batch_size=self.batchsize, shuffle=False, num_workers=4)
        num_classes = 100
        return trainset, testset, trainloader, testloader, num_classes

    def load_cifar10(self):
        CIFAR_MEAN = [0.4914, 0.4822, 0.4465]
        CIFAR_STD = [0.2470, 0.2435, 0.2616]
        normalize = T.Normalize(CIFAR_MEAN, CIFAR_STD)
        train_transform = T.Compose([T.RandomHorizontalFlip(), T.RandomCrop(32, padding=4), T.ToTensor(), normalize])
        test_transform = T.Compose([T.ToTensor(), normalize])
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=self.batchsize, shuffle=True, num_workers=4)
        testloader = torch.utils.data.DataLoader(testset, num_workers=4, batch_size=self.batchsize)
        num_classes = 10
        return trainset, testset, trainloader, testloader, num_classes

    def load_mnist(self):
        MNIST_MEAN = [0.1307]
        MNIST_STD = [0.3081]
        normalize = T.Normalize(MNIST_MEAN, MNIST_STD)
        train_transform = T.Compose([T.ToTensor(), normalize])
        test_transform = T.Compose([T.ToTensor(), normalize])
        trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=train_transform)
        testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=test_transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=self.batchsize, shuffle=True, num_workers=4)
        testloader = torch.utils.data.DataLoader(testset, num_workers=4, batch_size=self.batchsize)
        num_classes = 10
        return trainset, testset, trainloader, testloader, num_classes

    def load_tinyimagenet(self):
        imgnet_MEAN = [0.485, 0.456, 0.406]
        imgnet_STD = [0.229, 0.224, 0.225]
        normalize = T.Normalize(np.array(imgnet_MEAN) / 255, np.array(imgnet_STD) / 255)
        denormalize = T.Normalize(-np.array(imgnet_MEAN) / np.array(imgnet_STD), 255 / np.array(imgnet_STD))
        # train_transform = T.Compose([T.RandomHorizontalFlip(), T.Resize((32, 32), antialias=True), T.ToTensor(), normalize])
        # test_transform = T.Compose([T.ToTensor(), T.Resize((32, 32), antialias=True), normalize])
        train_transform = T.Compose([T.RandomHorizontalFlip(), T.ToTensor(), normalize])
        test_transform = T.Compose([T.ToTensor(), normalize])
        trainset = torchvision.datasets.ImageFolder(root = os.path.join(self.datasetdir,'train'), transform=train_transform)
        testset = torchvision.datasets.ImageFolder(root = os.path.join(self.datasetdir,'val'), transform=test_transform)

        num_classes = 200
        small_labels = {}
        with open(os.path.join(self.datasetdir, "words.txt"), "r") as dictionary_file:
            line = dictionary_file.readline()
            while line:
                label_id, label = line.strip().split("\t")
                small_labels[label_id] = label
                line = dictionary_file.readline()

        labels = {}
        label_ids = {}
        for label_index, label_id in enumerate(trainset.classes):
            label = small_labels[label_id]
            labels[label_index] = label
            label_ids[label_id] = label_index

        val_label_map = {}
        with open(os.path.join(self.datasetdir,"val","val_annotations.txt"), "r") as val_label_file:
            line = val_label_file.readline()
            while line:
                file_name, label_id, _, _, _, _ = line.strip().split("\t")
                val_label_map[file_name] = label_id
                line = val_label_file.readline()

        for i in range(len(testset.imgs)):
            file_path = testset.imgs[i][0]

            file_name = os.path.basename(file_path)
            label_id = val_label_map[file_name]

            testset.imgs[i] = (file_path, label_ids[label_id])

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=self.batchsize, shuffle=True, num_workers=4, pin_memory=True)
        testloader = torch.utils.data.DataLoader(testset, batch_size=self.batchsize, shuffle=False, num_workers=4, pin_memory=True)
        return trainset, testset, trainloader, testloader, num_classes
