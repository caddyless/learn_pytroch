import turicreate as tc
import os

source_dir = '/home/qualcomm'
step_length = 100
k = 2


def get_accuracy(path='.'):
    reference_data = tc.load_sframe(path + '/data.sframe')
    if os.path.isdir(path + '/savedmodel.model'):
        model = tc.load_model(path + '/savedmodel.model')
    else:
        model = tc.image_similarity.create(reference_data)
        model.save(path + '/savedmodel.model')
    correct = 0
    mistake = 0
    index = 0
    distance = 0
    while index < len(reference_data):
        if index + step_length < len(reference_data):
            query_results = model.query(
                reference_data[index:index + step_length], k=k, verbose=False)
            index += step_length
        else:
            query_results = model.query(
                reference_data[index:], k=k, verbose=False)
            index = len(reference_data)
        assert len(query_results) % k == 0, 'length error!'
        for i in range(len(query_results) / k):
            category = [reference_data[query_results[i * k + j]
                                       ['reference_label']]['path'][23:-16] for j in range(k)]
            if category[0] == category[1] or (
                    ('Faces' in category[0]) and (
                    'Face' in category[1])):
                correct += 1
            else:
                mistake += 1
            for j in range(k):
                distance += query_results[i * k + j]['distance']

        if (index + 1) % 1000 == 0:
            print(str(index + 1) + ' completed!')

    print('正确个数为:' + str(correct))
    print('错误个数为:' + str(mistake))
    print('正确率为:' + str(correct / (correct + mistake)))
    print('平均距离为： ' + str(distance / len(reference_data)))


if __name__ == '__main__':
    get_accuracy()
