import tensorflow as tf
import os
import time
from configuration import save_model_dir, test_image_dir
from train import get_model
from prepare_data import load_and_preprocess_image

os.environ['CUDA_VISIBLE_DEVICES']= '1'
def get_class_id(image_root):
    id_cls = {}
    dir=os.listdir(image_root)
    dir.sort()
    for i, item in enumerate(dir):
        if os.path.isdir(os.path.join(image_root, item)):
            id_cls[i] = item
    print('id_cls:{}'.format(id_cls))
    return id_cls


if __name__ == '__main__':
    try:
        # GPU settings
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)

        model = get_model()
        model.load_weights(filepath=save_model_dir)
        prob = {}
        timelist=[]
        id_cls = get_class_id("./original_dataset")
        for root, dirs, files in os.walk(test_image_dir):

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

                    print("The predicted category of \'{}\' is: {}".format(f, id_cls[idx]))

    finally:
        for i in prob:
            print(i, prob[i])

