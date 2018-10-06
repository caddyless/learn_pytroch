import scipy.misc as mc
import pickle
import os
import numpy as np

file_list = [
    'test_batch',
    'data_batch_1',
    'data_batch_2',
    'data_batch_3',
    'data_batch_4',
    'data_batch_5']
source_dir = r'./data/cifar-10-batches-py/'
class_name = [
    'airplane',
    'automobile',
    'bird',
    'cat',
    'deer',
    'dog',
    'frog',
    'horse',
    'ship',
    'truck']


def load_file(filename):
    with open(source_dir + 'batches.meta', 'rb') as fo:
        data = pickle.load(fo, encoding='latin1')
        for item in data['label_names']:
            if os.path.isdir(source_dir + 'pics/' + item):
                continue
            else:
                os.makedirs(source_dir + 'pics/' + item)
    class_num = np.zeros(10, dtype=np.uint16)
    for fn in filename:
        with open(source_dir + fn, 'rb') as fo:
            data = pickle.load(fo, encoding='latin1')
            image_set = data['data'].reshape(10000, 3, 32, 32)
            for i in range(10000):
                row = image_set[i]
                row = np.array(row, dtype=np.uint8)
                red = row[0].reshape(1024, 1)
                green = row[1].reshape(1024, 1)
                blue = row[2].reshape(1024, 1)
                img = np.hstack((red, green, blue))
                img = img.reshape(32, 32, 3)
                label = data['labels'][i]
                path = source_dir + 'pics/' + class_name[label] + '/image_'
                while os.path.isfile(path + str(class_num[label]) + '.png'):
                    class_num[label] += 1
                mc.imsave(path + str(class_num[label]) + '.png', img)
                if (i + 1) % 1000 == 0:
                    print(data['batch_label'] + '  ' +
                          str(i + 1) + ' completed!')


if __name__ == '__main__':
    load_file(file_list)
