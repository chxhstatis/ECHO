import tensorflow as tf
import os
import time
import csv
from configuration import save_model_dir, test_image_dir
from train import get_model
from prepare_data import load_and_preprocess_image

slide_dir=''
slide_list_raw=[]
slide_list=[]
dl_list=[]
def get_class_id(image_root):
    id_cls = {}
    dir = os.listdir(image_root)
    dir.sort()
    for i, item in enumerate(dir):
        if os.path.isdir(os.path.join(image_root, item)):
            id_cls[i] = item
    print('id_cls:{}'.format(id_cls))
    return id_cls


if __name__ == '__main__':
    for file in os.listdir(slide_dir):
        full_name=os.path.join(slide_dir,file)
        if os.path.isdir(full_name):
            if file[15]=='A':
                slide_list_raw.append(full_name)
    # print(slide_list_raw)
    with open('label_add0.csv') as f_csv:
        ff_csv=csv.reader(f_csv)
        for line in ff_csv:
            slide_name=line[0][:15]+'A'
            dl_list.append(slide_name)

    # print(dl_list)
    for raw_slide in slide_list_raw:
        if raw_slide.split('/')[-1][:16] in dl_list:
            slide_list.append(raw_slide)
    # print(slide_list)
    try:
        # GPU settings
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)

        model = get_model()
        model.load_weights(filepath=save_model_dir)
        timelist=[]
        for slide in slide_list:
            # print(slide)
            prob = {}
            num_dict={}
            id_cls = get_class_id("./original_dataset")
            for root, dirs, files in os.walk(slide):
                for f in files:
                    full_f = os.path.join(root, f)
                    if os.path.splitext(full_f)[-1]=='.jpeg':

                        image_raw = tf.io.read_file(filename=full_f)
                        image_tensor = load_and_preprocess_image(image_raw)
                        image_tensor = tf.expand_dims(image_tensor, axis=0)

                        pred = model(image_tensor, training=False)
                        idx = tf.math.argmax(pred, axis=-1).numpy()[0]
                        if not id_cls[idx] in prob:
                            prob[id_cls[idx]] = 1
                        else:
                            prob[id_cls[idx]] += 1

                        print("The WSI ({}) of \'{}\' is: {}".format(slide.split('/')[-1][:16],f, id_cls[idx]))
            with open('all_slide_eff.txt','a') as f:
                f.write(slide.split('/')[-1])
                for i in ['G0','G1','G2','G3','G4']:
                    if i in prob:
                        f.write(',')
                        f.write(str(prob[i]))
                    else:
                        f.write(',')
                        f.write('0')
                f.write('\n')





    finally:
        pass

