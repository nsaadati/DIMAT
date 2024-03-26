import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim

def _weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
#             self.shortcut = LambdaLayer(lambda x:
#                                         F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(planes)
            )


    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, w=1, num_classes=10, text_head=False):
        super(ResNet, self).__init__()
        self.in_planes = int(w*16)

        self.conv1 = nn.Conv2d(3, int(w*16), kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(int(w*16))
        self.layer1 = self._make_layer(block, int(w*16), num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, int(w*32), num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, int(w*64), num_blocks[2], stride=2)
        if text_head:
            num_classes = 512
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(int(w*64), num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out) # F.avg_pool2d(out, int(out.size()[3]))
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    def train_model(self, num_epochs, train_loader, device, optimizer_type, softmax=False):
        self.train()

        if optimizer_type == 'adam':
            optimizer = optim.Adam(self.parameters(), lr=0.001)
        elif optimizer_type == 'sgd':
            optimizer = optim.SGD(self.parameters(), lr=0.001, momentum=0.9)
        else:
            raise ValueError("Unsupported optimizer type")

        for epoch in range(num_epochs):  # Adjust the number of epochs as needed
            running_loss = 0.0
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = self(data)
                if softmax:
                    output = F.log_softmax(output, dim=1)
                loss = F.nll_loss(output, target)
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                if batch_idx % 10 == 9:    # Print the loss every 10 mini-batches
                    print(f'Epoch: {epoch + 1}, Batch: {batch_idx + 1}, Loss: {running_loss / 10:.3f}')
                    running_loss = 0.0
        print('Training complete')

        train_accuracy, tloss = self.test_accuracy(train_loader, device)
        print('Training Accuracy', train_accuracy)
        return train_accuracy, tloss

    def test_accuracy(self, dataloader, device, softmax=True):
        self.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in dataloader:
                data, target = data.to(device), target.to(device)
                output = self(data)
                if softmax:
                    output = F.log_softmax(output, dim=1)
                test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()


        test_loss /= len(dataloader.dataset)
        acc = 100. * correct / len(dataloader.dataset)

        return acc, test_loss  # Return both accuracy and average loss

def resnet20(w=8, num_classes=10, text_head=False):
    return ResNet(BasicBlock, [3, 3, 3], w=w, num_classes=num_classes, text_head=text_head)
