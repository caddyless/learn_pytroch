import turicreate as tc
import os

source_dir = '/home/qualcomm'


def get_accuracy(path='./'):
    reference_data = tc.load_sframe(path + 'data.sframe')
    if os.path.isdir(path + '/savedmodel.model'):
        model = tc.load_model(path + '/savedmodel.model')
    correct = 0
    mistake = 0
    for item in reference_data:
        query_results = model.query(item, k=2)
        path_list = [reference_data[result['reference_label']]['path'][23:-16]
                     for result in query_results]
        print(path_list)
        assert len(path_list) == 2, 'length of path_list error'
        if path_list[0] == path_list[1]:
            correct += 1
        else:
            mistake += 1
    print('正确个数为:' + str(correct))
    print('错误个数为:' + str(mistake))
    print('正确率为:' + str(correct / (correct + mistake)))


if __name__ == '__main__':
    get_accuracy()
