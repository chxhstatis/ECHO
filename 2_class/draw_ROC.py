import csv
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
import os
from configuration import model_index
from prepare_label import labelname

pos_label=0
def get_curve(file='index.csv'):
    scores = []
    labels = []
    with open(file) as f:
        f_csv = csv.reader(f)
        for line in f_csv:
            scores.append(float(line[1]))
            labels.append(float(line[0]))
    return np.array(labels), np.array(scores)


def draw_cure(pos_label=pos_label):
    labels, scores = get_curve()
    fpr, tpr, thresholds = metrics.roc_curve(labels, scores, pos_label)
    auc = metrics.auc(fpr, tpr)
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='blue',
             lw=lw, label='ROC curve (area = %0.2f)' % auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")

    plt.savefig("Result/{}-{}-ROC-10x.png".format(labelname,model_index))
    plt.show()

os.rename('index.txt', 'index.csv')
draw_cure()
