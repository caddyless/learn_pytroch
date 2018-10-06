import scipy.misc as mc
import pickle
import os
import numpy as np

filename = [
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


def load_file(fn=filename):
    with open(source_dir + 'batches.meta', 'rb') as fo:
        data = pickle.load(fo, encoding='latin1')
        for item in data['label_names']:
            if os.path.isdir(source_dir + 'pics/' + item):
                continue
            else:
                os.makedirs(source_dir + 'pics/' + item)
    for fn in filename:
        with open(source_dir + fn, 'rb') as fo:
            data = pickle.load(fo, encoding='latin1')
            class_num = np.zeros((1, 10), dtype=np.uint8)
            for i in range(len(data['data'])):
                row = data['data'][i]
                row = np.array(row, dtype=np.uint8)
                img = row.reshape(32, 32, 3)
                label = data['labels'][i]
                path = source_dir + 'pics/' + class_name[label] + '/image_'
                while os.path.isfile(path + str(class_num[label])):
                    class_num[label] += 1
                mc.imsave(path + str(class_num[label]), img)


if __name__ == '__main__':
    load_file()
