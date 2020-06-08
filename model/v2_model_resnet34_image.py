from common  import *
import pretrainedmodels

# -*- coding: utf-8 -*-
"""
@Time    : 2019/5/27 20:21
@Author  : QuYue
@File    : models.py
@Software: PyCharm
Introduction: Models for diagnosing the heart disease by the ECG.
"""
#%% Import Packages
import numpy as np
import torch
import torch.nn as nn
#%% Functions
class ResidualBlock(nn.Module):
    # Residual Block
    def __init__(self, in_channel, out_channel, stride=1, shortcut=None):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=(17, 1), stride=stride, padding=(8, 0), bias=False),
            nn.BatchNorm2d(num_features=out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=(17, 1), stride=1, padding=(8, 0), bias=False),
            nn.BatchNorm2d(num_features=out_channel)
        )
        self.right = shortcut  # sth like nn.Module
        # self.right = nn.Sequential(
        #     nn.Conv2d(in_channel, out_channel, kernel_size=(1,1), padding=0,stride=stride, bias=True),
        #     nn.BatchNorm2d(num_features=out_channel))

    def forward(self, x):
        out = self.left(x)
        residual = x if self.right is None else self.right(x)
        out += residual
        return nn.functional.relu(out)  # Relu(out + residual)

class Net(nn.Module):
    # ResNet34: including 5 layers
    def __init__(self, inplanes, planes):
        super(Net, self).__init__()
        '''
         {(input_height - kernel_size + 2*padding) / stride[0] }+1
         {(input_Weight - kernel_size + 2*padding) / stride[1] }+1
        '''
        # First convolution and pooling layer
        self.pre = nn.Sequential(
            # nn.Conv2d(12,12,(1,1),stride=1,padding=0,bias=False),
            # nn.BatchNorm2d(12),
            # nn.Dropout(0.7),
            # nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=inplanes, out_channels=64, kernel_size=(33, 1), stride=1, padding=(16,0), bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=False),
        )
        # nn.Conv2d(32, 32, (1, 1), stride=1, padding=0, bias=False),
        # nn.BatchNorm2d(32),
        # nn.Dropout(0.7),
        # nn.ReLU(inplace=True))

        # 4 layers : which include 4、4、4、4 Residual Blocks
        self.layer0 = self.make_layer(64, 64, 4, stride=2)
        self.layer1 = self.make_layer(64, 128, 4, stride=2)
        self.layer2 = self.make_layer(128, 192, 4, stride=2)
        self.layer3 = self.make_layer(192, 256, 4, stride=2)

        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_features=planes, out_features=9)

    def make_layer(self, in_channel, out_channel, num_blocks, stride=1):
        # create layers (include a few of block)
        shortcut0 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=(1, 1), stride=stride, bias=False),
            nn.BatchNorm2d(num_features=out_channel)
        )

        shortcut1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(15, 1), stride=2, padding=(7,0))
        )

        # kernel size=1
        layers = [ResidualBlock(in_channel, out_channel, stride, shortcut0)]  # first block with shortcut
        for i in range(1, num_blocks):
            if i == 2:
                layers.append(ResidualBlock(out_channel, out_channel, stride, shortcut=shortcut1))
            else:
                layers.append(ResidualBlock(out_channel, out_channel))  # other blocks without shortcut
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.pre(x)
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        output = self.fc(x)

        return output

#%% Main Function
def run_check_net():
    x = np.random.randn(1, 12, 72000, 1)
    x = torch.Tensor(x)
    resnet = Net(12, 256)
    output = resnet(x)

    print(output.shape)

def compute_beta_score(labels, output, beta, num_classes, check_errors=True):
    # Check inputs for errors.
    if check_errors:
        if len(output) != len(labels):
            raise Exception('Numbers of outputs and labels must be the same.')

    # Populate contingency table.
    num_recordings = len(labels)

    fbeta_l = np.zeros(num_classes)
    gbeta_l = np.zeros(num_classes)
    fmeasure_l = np.zeros(num_classes)
    accuracy_l = np.zeros(num_classes)

    f_beta = 0
    g_beta = 0
    f_measure = 0
    accuracy = 0

    # Weight function
    C_l = np.ones(num_classes);

    for j in range(num_classes):
        tp = 0
        fp = 0
        fn = 0
        tn = 0

        for i in range(num_recordings):

            num_labels = np.sum(labels[i])

            if labels[i][j] and output[i][j]:
                tp += 1 / num_labels
            elif not labels[i][j] and output[i][j]:
                fp += 1 / num_labels
            elif labels[i][j] and not output[i][j]:
                fn += 1 / num_labels
            elif not labels[i][j] and not output[i][j]:
                tn += 1 / num_labels

        # Summarize contingency table.
        if ((1 + beta ** 2) * tp + (fn * beta ** 2) + fp):
            fbeta_l[j] = float((1 + beta ** 2) * tp) / float(((1 + beta ** 2) * tp) + (fn * beta ** 2) + fp)
        else:
            fbeta_l[j] = 1.0

        if (tp + fp + beta * fn):
            gbeta_l[j] = float(tp) / float(tp + fp + beta * fn)
        else:
            gbeta_l[j] = 1.0

        if tp + fp + fn + tn:
            accuracy_l[j] = float(tp + tn) / float(tp + fp + fn + tn)
        else:
            accuracy_l[j] = 1.0

        if 2 * tp + fp + fn:
            fmeasure_l[j] = float(2 * tp) / float(2 * tp + fp + fn)
        else:
            fmeasure_l[j] = 1.0

    for i in range(num_classes):
        f_beta += fbeta_l[i] * C_l[i]
        g_beta += gbeta_l[i] * C_l[i]
        f_measure += fmeasure_l[i] * C_l[i]
        accuracy += accuracy_l[i] * C_l[i]

    f_beta = float(f_beta) / float(num_classes)
    g_beta = float(g_beta) / float(num_classes)
    f_measure = float(f_measure) / float(num_classes)
    accuracy = float(accuracy) / float(num_classes)

    return accuracy, f_measure, f_beta, g_beta

#https://stackoverflow.com/questions/43162506/undefinedmetricwarning-f-score-is-ill-defined-and-being-set-to-0-0-in-labels-wi
def metric(truth, predict):
    truth_for_cls = np.sum(truth, axis=0) + 1e-11
    predict_for_cls = np.sum(predict, axis=0) + 1e-11

    # TP
    count = truth + predict
    count[count != 2] = 0
    TP = np.sum(count, axis=0) / 2

    precision = TP / predict_for_cls
    recall = TP / truth_for_cls

    return precision, recall


# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    # run_check_basenet()
    run_check_net()


    print('\nsuccess!')


