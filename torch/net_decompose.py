from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import get_image
import matplotlib.pyplot as plt
import numpy as np
import torch


classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
d_pie1 = 3
d_pie2 = 8


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5*5 square convolution
        # kernel

        self.conv1W = nn.Conv2d(3, 3, 5,bias=True)
        self.conv1P=nn.Conv2d(3,6,1,bias=True)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2W = nn.Conv2d(6, 8, 5,bias=True)
        self.conv2P=nn.Conv2d(8,16,1,bias=True)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.conv1W(x)
        x = self.conv1P(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.conv2W(x)
        x = self.conv2P(x)
        x = F.relu(x)
        x = self.pool(x)
        # x = self.pool(F.relu(self.conv1(x)))
        # x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def train_net(trainloader, net):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    for epoch in range(2):  # loop over the dataset multiple times

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
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training')


def test(testloader):
    dateiter = iter(testloader)
    images, labels = dateiter.next()

    # print images
    imshow(torchvision.utils.make_grid(images))
    print('GroundTruth: ', ' '.join('%5s' %
                                    classes[labels[j]] for j in range(4)))
    outputs = net(Variable(images))
    predicted = torch.max(outputs.data, 1)
    print('Predicted: ', ' '.join('%5s' % classes[predicted[1][j]]
                                  for j in range(4)))


if __name__ == '__main__':
    net = Net()
    trainloader, testloader = get_image.download_img()
    train_net(trainloader, net)
