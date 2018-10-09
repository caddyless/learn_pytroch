import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from entry import get_parameter

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.figure('image')
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def download_img(model='train'):
    parameter=get_parameter()
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=parameter['batch_size'],
                                              shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=parameter['batch_size'],
                                             shuffle=False, num_workers=2)
    samples = torch.utils.data.RandomSampler(trainset, num_samples=3000)
    # get some random training images
    # dataiter = iter(trainloader)
    # images, labels = dataiter.next()
    #
    # # show images
    # imshow(torchvision.utils.make_grid(images))
    # # print labels
    # print(' '.join('%5s' % classes[labels[j]] for j in range(4)))
    if model=='train':
        return trainloader, testloader
    if model=='sample':
        return samples
    else:
        print('argument error')
        return


if __name__ == '__main__':
    # download cifar10
    trainset, testset = download_img('train')
