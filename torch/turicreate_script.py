import turicreate as tc
import os

source_dir = '/home/qualcomm/new_seg_images'
origin_dir = '/home/qualcomm/101_ObjectCategories'


def create_mod(path):
    if os.path.isfile(path + '/data.sframe'):
        print('reference_data is existed')
        reference_data = turicreate.load_sframe(path + '/data.sframe')
    else:
        # Load images from the downloaded data
        reference_data = tc.image_analysis.load_images(path)
        reference_data = reference_data.add_row_number()
    if os.path.isfile(path + '/savedmodel.model'):
        print('mod is existed')
        model = turicreate.load_model(path + '/savedmodel.model')
    else:
        # Save the SFrame for future use
        reference_data.save(path + '/data.sframe')
        model = tc.image_similarity.create(reference_data)
        model.save(path + '/savedmodel.model')
    return reference_data, model


def compare_img(origin_results,current_results):
    o=origin_results
    c=current_results
    assert len(o)==len(c) ,'len(origin_results) are not equal to len(current_results)'


def query(path,start=0,end=10,k=10):
    reference_data,model=create_mod(path)
    query_results = model.query(reference_data[0:10], k=10)
    return query_results


if __name__ == '__main__':
    turicreate.config.set_num_gpus(-1)
    origin_results=query(origin_dir)
    folders = os.listdir(source_dir)
    for folder in folders:
        path = os.path.join(source_dir, folder)
        current_results=query(path)
        compare_img(origin_results,current_results)
