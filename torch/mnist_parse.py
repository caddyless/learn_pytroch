import os
import struct
import numpy as np
import scipy.misc


def load_mnist(path, kind='train'):
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels.idx1-ubyte'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images.idx3-ubyte'
                               % kind)
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II',
                                 lbpath.read(8))
        labels = np.fromfile(lbpath,
                             dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII',
                                               imgpath.read(16))
        images_set = np.fromfile(imgpath,
                                 dtype=np.uint8).reshape(len(labels), 784)
        images = []
        for rows in images_set:
            img = rows.reshape(28, 28)
            images.append(img)
        # images = [rows.reshape(28, 28) for rows in images_set]
        for i in range(10):
            if os.path.isdir(path + '/' + str(i)):
                continue
            else:
                os.makedirs(path + '/' + str(i))
        for i in range(len(images)):
            scipy.misc.imsave(
                r'{root}/{label}/image_{num}.jpg'.format(
                    root=path, label=str(
                        labels[i]), num=str(i)), images[i])
            if (i + 1) % 1000 == 0:
                print('已完成' + str(i+1) + '张')
    return images, labels


if __name__ == '__main__':
    load_mnist(path='./data')
