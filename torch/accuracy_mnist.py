import turicreate as tc
import os
import time
import numpy as np

source_dir = './mnist'
step_length = 100
k = 2
categories=['0','1','2','3','4','5','6','7','8','9']


def get_accuracy(path=source_dir):
    if os.path.isdir(path + '/data.sframe'):
        print('reference_data is existed')
        reference_data = tc.load_sframe(path + '/data.sframe')
    else:
        # Load images from the downloaded data
        reference_data = tc.image_analysis.load_images(path)
        reference_data = reference_data.add_row_number()
        reference_data.save(path + '/data.sframe')
    if os.path.isdir(path + '/savedmodel.model'):
        model = tc.load_model(path + '/savedmodel.model')
    else:
        model = tc.image_similarity.create(reference_data)
        model.save(path + '/savedmodel.model')
    correct = np.zeros(10,dtype=np.uint16)
    mistake = np.zeros(10,dtype=np.uint16)
    index = 0
    distance = np.zeros(10,dtype=np.uint16)
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
        for i in range(int(len(query_results) / k)):
            category = [reference_data[query_results[i * k + j]
                                       ['reference_label']]['path'].split('/')[2] for j in range(k)]
            if category[0] == category[1] or (
                    ('Faces' in category[0]) and (
                    'Face' in category[1])):
                
                correct[categories.index(category[0])] += 1
            else:
                mistake[categories.index(category[0])] += 1           
            distance[categories.index(category[0])] += query_results[i * k + 1]['distance']

        print(str(index) + ' completed!')

    print('正确个数为:' + str(correct))
    print('错误个数为:' + str(mistake))
    print('正确率为:' + str([correct[i]/(correct[i]+mistake[i]) for i in range(categories)])
    print('平均距离为：' + str([distance[i]/(correct[i]+mistake[i]) for i in range(categories)])


if __name__ == '__main__':
    start=time.time()
    get_accuracy()
    end=time.time()
    time_cost=end-start
    format_time=str(int(time_cost/3600))+'h'+str(int((time_cost%3600)/60))+'m'
    print('total time used:'+format_time)
