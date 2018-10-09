import torch
import torchvision as tv
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import get_image
from entry import get_parameter
import decompose

# get parameter from a entry.py
p = get_parameter()
# # get samples from feature map of conv1 and conv2
# sample_y1 = np.zeros((p['c2'], p['sample_num']))
# sample_y2 = np.zeros((p['c3'], p['sample_num']))
# flag of extract model
EXTRACT = False


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # c_in:image channel  c2:output channel of conv1  c3:output channel of conv2
        # k: the size of filter
        self.conv1 = nn.Conv2d(p['c_in'], p['c2'], p['k'])
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(p['c2'], p['c3'], p['k'])
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(p['c3'] * p['k'] * p['k'], 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.conv1(x)
        if EXTRACT:
            sample_y1 = x[0]
        x = F.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        if EXTRACT:
            sample_y2 = x[0]
        x = F.relu(x)
        x = self.pool(x)
        # x = self.pool(F.relu(self.conv1(x)))
        # x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, p['c3'] * p['k'] * p['k'])
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        if EXTRACT:
            return x, sample_y1, sample_y2
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


def train_net(trainloader, net, epoch, max, padding=1000):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    for e in range(epoch):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data

            # wrap them in Variable
            inputs, labels = Variable(inputs), Variable(labels)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.data
            if (i + 1) % padding == 0:  # print every padding mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch, i + 1, running_loss / padding))
                running_loss = 0.0
            if not max==None:
                if (i + 1) > max:
                    break


if __name__ == '__main__':
    net = Net()
    trainloader, testloader = get_image.download_img('train')
    train_net(trainloader, net, epoch=1, max=None)
    print('train finish!')
    print('feature extract begin!')
    EXTRACT = True
    dateiter = iter(testloader)
    images, labels = dateiter.next()
    outputs, sample_y1, sample_y2 = net(images)
    EXTRACT = False
    params = net.state_dict()
    weight = params['conv1.weight']
    bias = params['conv1.bias']
    sy1 = sample_y1.data.numpy()
    decompose.feature_decompose(sy1, weight, bias, p['c2'])
