from AF.dataset_af_full_size import *


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, downsample=None,
                 dropout_rate = 0.5):
        super(BasicBlock, self).__init__()
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.stride = stride
        self.kernel_size = kernel_size
        self.bn1 = nn.BatchNorm1d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.conv1 = nn.Conv1d(in_planes, out_planes, kernel_size=self.kernel_size, stride=self.stride,
                     padding=int(self.kernel_size / 2), bias=False)
        self.bn2 =  nn.BatchNorm1d(out_planes)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.conv2 = nn.Conv1d(out_planes, out_planes, kernel_size=self.kernel_size, stride=1,
                     padding=int(self.kernel_size / 2))
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.dropout1(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout2(out)
        out = self.conv2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class Net(nn.Module):

    def __init__(self, in_planes, num_classes=1000, kernel_size=15, dropout_rate = 0.5):
        super(Net, self).__init__()
        self.dilation = 1
        self.in_planes = in_planes
        self.out_planes = 64
        self.stride = 1
        self.kernel_size = kernel_size

        # pre conv
        self.conv1 = nn.Conv1d(self.in_planes, self.out_planes, kernel_size=self.kernel_size, stride=1, padding=int(self.kernel_size/2),
                               bias=False)
        self.in_planes = self.out_planes
        self.bn1 =  nn.BatchNorm1d(self.out_planes)
        self.relu = nn.ReLU(inplace=True)

        # first block
        self.conv2 = nn.Conv1d(self.out_planes, self.out_planes, kernel_size=self.kernel_size, stride=2, padding=int(self.kernel_size/2),
                               bias=False)
        self.bn2 = nn.BatchNorm1d(self.out_planes)
        self.dropout = nn.Dropout(dropout_rate)
        self.conv3 = nn.Conv1d(self.out_planes, self.out_planes, kernel_size=self.kernel_size, stride=1, padding=int(self.kernel_size/2),
                               bias=False)
        self.maxpool = nn.MaxPool1d(kernel_size=self.kernel_size, stride=2, padding=int(self.kernel_size/2))

        layers = []
        for i in range(1, 16):
            if i % 4 == 0 :
                self.out_planes = self.in_planes + 64

            if i % 4 == 0 :
                downsample = nn.Sequential(
                    nn.Conv1d(self.in_planes, self.out_planes, kernel_size=1, stride=2, bias=False),
                    nn.BatchNorm1d(self.out_planes))
                self.stride = 2
            elif i % 2 == 0 :
                downsample = self.maxpool
                self.stride = 2
            else :
                downsample = None
                self.stride = 1

            layers.append(BasicBlock(self.in_planes, self.out_planes, self.kernel_size, self.stride, downsample))

            self.in_planes = self.out_planes

        self.layers = nn.Sequential(*layers)

        self.bn3 = nn.BatchNorm1d(self.out_planes)
        self.lstm = nn.LSTM(118, 118, bidirectional = True)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(self.out_planes, num_classes)

    def forward(self, x):
        # pre conv
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        # first block
        identity = x
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv3(x)
        x = x + self.maxpool(identity)

        # res block x 15
        x = self.layers(x)

        x = self.bn3(x)
        x = self.relu(x)
        x_shape = x.shape
        hidden = (torch.zeros(2, x_shape[1], x_shape[2]).cuda(), torch.zeros(2, x_shape[1], x_shape[2]).cuda())

        # x = x.permute(0, 2, 1)
        x, (hn, cn) = self.lstm(x, hidden)
        # x = x.permute(0, 2, 1)

        # print(x.shape)
        x = self.avgpool(x)
        # print(x.shape)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

def run_check_net():
    net = Net(12, num_classes=9, dropout_rate = 0.5)

    input = torch.randn(20, 12, 256*5)
    output = net(input)

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
