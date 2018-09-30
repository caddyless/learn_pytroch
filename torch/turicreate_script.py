import turicreate as tc
import os

source_dir='./new_seg_images'
turicreate.config.set_num_gpus(-1)

def create_mod(dir):
    path=os.path.join(source_dir,dir)
    if os.path.isfile(path+'/caltech-101.sframe'):
        print('mod is existed')
        return
    # Load images from the downloaded data
    reference_data = tc.image_analysis.load_images(path)
    reference_data = reference_data.add_row_number()

    # Save the SFrame for future use
    reference_data.save(path+'/caltech-101.sframe')
    model = tc.image_similarity.create(reference_data)
    model.save(path+'/savedmodel.model')
    return

def compare_img(folder):
    path=os.path.join(source_dir,folder)
    loaded_model = turicreate.load_model(path+'/savedmodel.model')



if __name__ == '__main__':
    folders=os.listdir(source_dir)
    for folder in folders:
        data=create_mod(folder)
        compare_img(folder)