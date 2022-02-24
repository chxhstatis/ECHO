#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:Mayako
import os
import csv
import shutil
import random

from configuration import model_index

labelname = 'grade_original'
wrong_label = ['NA', 'not reported', 'GX']
file_csv = 'label_add0.csv'
effi=1

class prepare_data():
    def __init__(self, label, wrong_label=None):
        self.wrong_label = wrong_label
        self.label = label
        self.original_dataset = os.path.join(os.getcwd(), 'original_dataset')
        self.dataset = ''

    # label_list is all labels in this variable, and column is the column where it is located
    def get_label(self):
        label_list = []
        index = 0
        column = 0
        with open(file_csv) as f:
            f_csv = csv.reader(f)

            for line in f_csv:
                if index == 0:
                    column = int(line.index(self.label))
                if not line[column] in label_list:
                    label_list.append(line[column])
                    index += 1
                # print(line)

        label_list.pop(0)
        for i in self.wrong_label:
            if i in label_list:
                label_list.remove(i)

        return label_list, column

    def creat_dir(self):
        label_list, column = self.get_label()
        original_dataset = self.original_dataset
        for label in label_list:
            if not os.path.exists(os.path.join(original_dataset, label)):
                os.mkdir(os.path.join(original_dataset, label))

    # Returns a dict containing the labels , and obtains the proportion of each label to balance the dataset
    def get_file_label(self):
        label_list, column = self.get_label()
        label_dir_temp = {}  # Labels corresponding to each WSI
        label_dir = {}
        classes = {}  # quantity per label
        proportion = {}
        index = 0
        min = 10000

        for label in label_list:
            classes[label] = 0
            proportion[label] = 0

        with open(file_csv) as f:
            f_csv = csv.reader(f)
            for line in f_csv:
                if not index == 0:
                    label_dir_temp[line[0]] = line[column]
                if line[column] in label_list:
                    classes[line[column]] += 1
                index += 1

        # the smallest class
        for c in classes:
            if classes[c] < min:
                min = classes[c]

        for l in label_list:
            proportion[l] = float(min / classes[l])

        # delete wrong_label
        for lab in label_dir_temp:
            if label_dir_temp[lab] in label_list:
                label_dir[lab] = label_dir_temp[lab]

        print(classes)
        print(proportion)

        return label_dir, proportion

    def copy_file(self):
        slide_list = []
        index = 0
        for i in os.listdir(self.dataset):
            if os.path.isdir(os.path.join(self.dataset, i)):
                slide_list.append(i[:-6])
                # slide_list.append(i)
        label_dir, proportion = self.get_file_label()
        file_list = []
        # print(slide_list)

        for label in label_dir:
            # print(label)
            # print(label,label_dir[label])
            if label[:-4] in slide_list:
                full_path = self.dataset + os.sep + label[:-4] + '_files'
                for root, dirs, files in os.walk(full_path):
                    for file in files:
                        if os.path.splitext(os.path.join(root, file))[-1] == '.jpeg':
                            file_list.append(os.path.join(root, file))

                random.shuffle(file_list)
                num = int(len(file_list) * proportion[label_dir[label]]*effi)
                for file in file_list[:num]:
                    dst = self.original_dataset + os.sep + label_dir[label] + os.sep + str(index) + '_' + \
                          os.path.split(file)[-1]
                    shutil.copyfile(file, dst)
                    print('copy {} to {}'.format(file, dst))
            file_list = []
            index += 1

    def create_model_dir(self):
        save_model_dir = "MyModel/labels/{}/{}".format(labelname, str(model_index))
        if not os.path.exists('MyModel/labels/{}'.format(labelname)):
            os.mkdir('MyModel/labels/{}'.format(labelname))
        if not os.path.exists(save_model_dir):
            os.mkdir(save_model_dir)

    def run_py(self):
        os.system(r'chmod a+x split_dataset.py ; ./split_dataset.py')
        os.system(r'chmod a+x to_tfrecord.py ; ./to_tfrecord.py')
        # os.system(r'chmod a+x train.py ; ./train.py')

    def clear_up(self):
        check = input('clear up?(y/n): \n')
        if check == 'y':
            if not os.path.exists('original_dataset'):
                os.mkdir('original_dataset')
            if not os.path.exists('dataset'):
                os.mkdir('dataset')
            else:
                shutil.rmtree('original_dataset')
                shutil.rmtree('dataset')
                os.mkdir('dataset')
                os.mkdir('original_dataset')


if __name__ == '__main__':
    start = prepare_data(labelname, wrong_label=wrong_label)
    start.clear_up()

    start.get_file_label()

    whether = input('Continue? (y/n): \n')
    if whether == 'y':
        start.creat_dir()
        start.create_model_dir()
        start.copy_file()
        start.run_py()
    else:
        print('exit.')
        pass

