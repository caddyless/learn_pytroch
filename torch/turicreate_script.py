import turicreate as tc
import os

source_dir='./new_seg_images'
turicreate.config.set_num_gpus(-1)

def create_mod(dir):
    path=os.path.join(source_dir,dir)
    if os.path.isfile(path+'/data.sframe'):
        print('reference_data is existed')
    else:
        # Load images from the downloaded data
        reference_data = tc.image_analysis.load_images(path)
        reference_data = reference_data.add_row_number()
    if os.path.isfile(path+'/savedmodel.model'):
        print('mod is existed')
    else:
        # Save the SFrame for future use
        reference_data.save(path + '/data.sframe')
        model = tc.image_similarity.create(reference_data)
        model.save(path + '/savedmodel.model')
    return

def compare_img(folder):
    path=os.path.join(source_dir,folder)
    loaded_model = turicreate.load_model(path+'/savedmodel.model')
    reference_data=turicreate.load_sframe(path+'/data.sframe')
    query_results = loaded_model.query(reference_data[0:10], k=10)
    query_results.head()



if __name__ == '__main__':
    folders=os.listdir(source_dir)
    for folder in folders:
        data=create_mod(folder)
        compare_img(folder)