# some training parameters

model_index = 1

from prepare_label import labelname
import os

EPOCHS = 80
BATCH_SIZE = 80
NUM_CLASSES = len(os.listdir('original_dataset'))
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
CHANNELS = 3
LEARNING_RATE = 0.0005

original_dataset = "original_dataset"
save_model_dir = "MyModels/{}/{}/{}/".format('20x',labelname, str(model_index))
save_every_n_epoch = 1
test_image_dir = ''

dataset_dir = "dataset/"
train_dir = dataset_dir + "train"
valid_dir = dataset_dir + "valid"
test_dir = dataset_dir + "test"
train_tfrecord = dataset_dir + "train.tfrecord"
valid_tfrecord = dataset_dir + "valid.tfrecord"
test_tfrecord = dataset_dir + "test.tfrecord"
# VALID_SET_RATIO = 1 - TRAIN_SET_RATIO - TEST_SET_RATIO
TRAIN_SET_RATIO = 0.6
TEST_SET_RATIO = 0.2

# choose a network
# 1: efficient_net_b0
# 2: ShuffleNetV2 1.0x
# 3: ResNet_18

