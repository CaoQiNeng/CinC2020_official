from common import *
from scipy import signal

#--------------
DATA_DIR = DATA_ROOT_PATH + '/CinC2020_V1'
# ['AF', 'I-AVB', 'LBBB', 'Normal', 'PAC',  'PVC', 'RBBB', 'STD', 'STE']
temp_class_map = pd.read_csv(DATA_DIR + '/evaluation-2020-master/dx_mapping_scored.csv')['SNOMED CT Code'].tolist()
unscored_map = pd.read_csv(DATA_DIR + '/evaluation-2020-master/dx_mapping_unscored.csv')['SNOMED CT Code'].tolist()
same_class = {'713427006' : '59118001', '63593006' : '284470004', '17338001' : '427172004'}
class_map = []

for i in range(len(temp_class_map)):
    if temp_class_map[i] not in same_class.values():
        class_map.append(str(temp_class_map[i]))

class_map = np.array(class_map)

class CinCDataset(Dataset):
    def __init__(self, split, mode, csv):
        self.split   = split
        self.csv     = csv
        self.mode    = mode

        df = pd.read_csv(DATA_DIR + '/SNOMED_overall_label.csv') #.fillna('')

        if split is not None:
            all_data = df['Recording'].values
            s = np.load(DATA_DIR + '/split_5F_v0/%s' % split)
            s_data = []
            for d in all_data:
                if d[:] in s:
                    s_data.append(d)

            df = df_loc_by_list(df, 'Recording', s_data)

        self.Recording = df['Recording'].values

        if mode == 'train' :
            truth_path = DATA_DIR + '/truth_%s_train' % split
            predict_path = DATA_DIR + '/predict_%s_train' % split
        else:
            truth_path = DATA_DIR + '/truth_%s_valid' % split
            predict_path = DATA_DIR + '/predict_%s_valid' % split

        os.makedirs(truth_path, exist_ok=True)
        os.makedirs(predict_path, exist_ok=True)

        for i, r in enumerate(self.Recording):
            shutil.copy(DATA_DIR + '/overall_hea/%s.hea' % r, truth_path + '/%s.hea' % r)
            shutil.copy(DATA_DIR + '/overall_hea/%s.hea' % r, predict_path + '/%s.csv' % r)

        normal = '426783006'

        # Find the label and output files.
        print('Finding label and output files...')
        label_files, output_files = find_challenge_files(truth_path, predict_path)

        # Load the labels and outputs.
        print('Loading labels and outputs...')
        label_classes, labels = load_labels(label_files, normal)

        # self.labels = []
        # df = df.set_index('Recording')
        # df = df.fillna(0)
        #
        # for i in range(len(df)) :
        #     d = df.loc[self.Recording[i]].tolist()[:-1]
        #
        #     label = np.zeros(len(unscored_map))
        #     for j in range(len(d)):
        #         if d[j] in same_class.keys():
        #             d[j] = same_class[d[j]]
        #
        #         l_index = np.where(class_map == int(d[j]))
        #         if len(l_index[0]) != 0:
        #             label[l_index[0][0]] = 1
        #
        #     self.labels.append(label)

        self.num_image = len(df)

    def __str__(self):
        string  = ''
        string += '\tlen = %d\n'%len(self)
        string += '\n'
        string += '\tmode     = %s\n'%self.mode
        string += '\tsplit    = %s\n'%self.split
        string += '\tcsv      = %s\n'%str(self.csv)
        string += '\tnum_image = %d\n'%self.num_image
        return string

    def __len__(self):
        return self.num_image

    def __getitem__(self, index):
        ecg_id = self.Recording[index]

        # label = self.labels[index]
        normal = '426783006'
        label_class, label_in_infor = load_labels([DATA_DIR + '/overall_hea/%s.hea' % ecg_id], normal)
        label = np.zeros(len(class_map))
        for j in range(len(label_class)):
            if label_class[j] in same_class.keys():
                label_class[j] = same_class[label_class[j]]

            l_index = np.where(class_map == label_class[j])
            if len(l_index[0]) != 0 and label_in_infor[j]:
                label[l_index[0][0]] = 1

        old_temp_ecg = sio.loadmat(DATA_DIR + '/overall/%s.mat' % ecg_id)['val']
        old_temp_ecg = np.array(old_temp_ecg / 1000)
        if ecg_id.startswith('S'):
            temp_ecg = []
            for i in range(len(old_temp_ecg)):
                temp_ecg.append(resample(old_temp_ecg[i,:], 1000))
        else:
            temp_ecg = []
            for i in range(len(old_temp_ecg)):
                temp_ecg.append(resample(old_temp_ecg[i,:], 500))

        temp_ecg = np.array(temp_ecg)

        ecg = np.zeros((12, 18000), dtype=np.float32)
        ecg[:,-temp_ecg.shape[1]:] = temp_ecg[:,-18000:]

        infor = Struct(
            index  = index,
            ecg_id = ecg_id,
            path = DATA_DIR + '/overall_hea/%s.hea' % ecg_id,
            label_class = label_class,
            label_in_infor = label_in_infor
        )

        return ecg, label, infor

class CustomSampler(Sampler):

    def __init__(self, dataset):
        self.dataset = dataset

        label = self.dataset.First_label

        self.AF_index  = np.where(label == 0)[0]
        self.I_AVB_index = np.where(label == 1)[0]
        self.LBBB_index = np.where(label == 2)[0]
        self.Normal_index = np.where(label == 3)[0]
        self.PAC_index = np.where(label == 4)[0]
        self.PVC_index = np.where(label == 5)[0]
        self.RBBB_index = np.where(label == 6)[0]
        self.STD_index = np.where(label == 7)[0]
        self.STE_index = np.where(label == 8)[0]

        #assume we know neg is majority class
        num_RBBB = len(self.RBBB_index)
        self.length = 9 * num_RBBB

    def __iter__(self):
        RBBB = self.RBBB_index.copy()
        np.random.shuffle(RBBB)
        # num_RBBB = len(self.RBBB_index)

        AF = np.random.choice(self.AF_index, len(self.AF_index), replace=False)
        I_AVB = np.random.choice(self.I_AVB_index, len(self.I_AVB_index), replace=False)
        LBBB = np.random.choice(self.LBBB_index, len(self.PAC_index), replace=True)
        Normal = np.random.choice(self.Normal_index, len(self.Normal_index), replace=False)
        PAC = np.random.choice(self.PAC_index, len(self.PAC_index), replace=True)
        PVC = np.random.choice(self.PVC_index, len(self.PVC_index), replace=False)
        STD = np.random.choice(self.STD_index, len(self.PAC_index), replace=False)
        STE = np.random.choice(self.STE_index, len(self.STD_index), replace=True)

        l = np.hstack([AF,I_AVB,LBBB,Normal,PAC,PVC,RBBB,STD,STE])
        np.random.shuffle(l)

        return iter(l)

    def __len__(self):
        return self.length

class BalanceClassSampler(Sampler):

    def __init__(self, dataset):
        self.dataset = dataset

        label = self.dataset.First_label

        self.AF_index  = np.where(label == 0)[0]
        self.I_AVB_index = np.where(label == 1)[0]
        self.LBBB_index = np.where(label == 2)[0]
        self.Normal_index = np.where(label == 3)[0]
        self.PAC_index = np.where(label == 4)[0]
        self.PVC_index = np.where(label == 5)[0]
        self.RBBB_index = np.where(label == 6)[0]
        self.STD_index = np.where(label == 7)[0]
        self.STE_index = np.where(label == 8)[0]

        #assume we know neg is majority class
        num_RBBB = len(self.RBBB_index)
        self.length = 9 * num_RBBB

    def __iter__(self):
        RBBB = self.RBBB_index.copy()
        np.random.shuffle(RBBB)
        num_RBBB = len(self.RBBB_index)

        AF = np.random.choice(self.AF_index, num_RBBB, replace=True)
        I_AVB = np.random.choice(self.I_AVB_index, num_RBBB, replace=True)
        LBBB = np.random.choice(self.LBBB_index, num_RBBB, replace=True)
        Normal = np.random.choice(self.Normal_index, num_RBBB, replace=True)
        PAC = np.random.choice(self.PAC_index, num_RBBB, replace=True)
        PVC = np.random.choice(self.PVC_index, num_RBBB, replace=True)
        STD = np.random.choice(self.STD_index, num_RBBB, replace=True)
        STE = np.random.choice(self.STE_index, num_RBBB, replace=True)

        l = np.stack([AF,I_AVB,LBBB,Normal,PAC,PVC,RBBB,STD,STE]).T
        l = l.reshape(-1)

        return iter(l)

    def __len__(self):
        return self.length

# Find Challenge files.
def find_challenge_files(label_directory, output_directory):
    label_files = list()
    output_files = list()
    for f in sorted(os.listdir(label_directory)):
        F = os.path.join(label_directory, f) # Full path for label file
        if os.path.isfile(F) and F.lower().endswith('.hea') and not f.lower().startswith('.'):
            root, ext = os.path.splitext(f)
            g = root + '.csv'
            G = os.path.join(output_directory, g) # Full path for corresponding output file
            if os.path.isfile(G):
                label_files.append(F)
                output_files.append(G)
            else:
                raise IOError('Output file {} not found for label file {}.'.format(g, f))

    if label_files and output_files:
        return label_files, output_files
    else:
        raise IOError('No label or output files found.')

# Load labels from header/label files.
def load_labels(label_files, normal):
    # The labels should have the following form:
    #
    # Dx: label_1, label_2, label_3
    #
    num_recordings = len(label_files)

    # Load diagnoses.
    tmp_labels = list()
    for i in range(num_recordings):
        with open(label_files[i], 'r') as f:
            for l in f:
                if l.startswith('#Dx'):
                    dxs = [arr.strip() for arr in l.split(': ')[1].split(',')]
                    tmp_labels.append(dxs)

    # Identify classes.
    classes = set.union(*map(set, tmp_labels))
    if normal not in classes:
        classes.add(normal)
        print('- The normal class {} is not one of the label classes, so it has been automatically added, but please check that you chose the correct normal class.'.format(normal))
    classes = sorted(classes)
    num_classes = len(classes)

    # Use one-hot encoding for labels.
    labels = np.zeros((num_recordings, num_classes), dtype=np.bool)
    for i in range(num_recordings):
        dxs = tmp_labels[i]
        for dx in dxs:
            j = classes.index(dx)
            labels[i, j] = 1

    # If the labels for the normal class and one or more other classes are positive, then make the label for the normal class negative.
    # If the labels for all classes are negative, then make the label for the normal class positive.
    normal_index = classes.index(normal)
    for i in range(num_recordings):
        num_positive_classes = np.sum(labels[i, :])
        if labels[i, normal_index]==1 and num_positive_classes>1:
            labels[i, normal_index] = 0
        elif num_positive_classes==0:
            labels[i, normal_index] = 1

    return classes, labels

def resample(data, sampr, after_Hz=300):
    data_len = len(data)
    propessed_data = signal.resample(data, int(data_len * (after_Hz / sampr)))

    return propessed_data

def null_collate(batch):
    batch_size = len(batch)

    input = []
    label = []
    infor = []
    for b in range(batch_size):
        input.append(batch[b][0])
        label.append(batch[b][1])
        infor.append(batch[b][-1])

    label = torch.from_numpy(np.stack(label)).float()
    input = torch.from_numpy(np.stack(input)).float()

    return input, label, infor

def run_check_DataLoader():
    batch_size = 72
    train_dataset = CinCDataset(
        mode='train',
        csv='train.csv',
        split='valid_a%d_687.npy' % 0,
    )
    train_loader = DataLoader(
        train_dataset,
        # sampler     = RandomSampler(train_dataset),
        shuffle = True,
        batch_size=batch_size,
        drop_last=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=null_collate
    )

    label_save_path = 'F:/data_root/CinC2020/check_label'
    ecg_save_path = 'F:/data_root/CinC2020/check_ecg'

    a = 0
    for t, (input, truth, infor) in enumerate(train_loader):
        print(infor.ecg_id)
        print(truth)
        # for i in range(len(infor)):
        #     with open(label_save_path + '/' + infor[i].ecg_id + '.json', 'w') as f:
        #         json_str = json.dumps({'label' : truth[i].tolist()})
        #         f.write(json_str)
        #
        #     sio.savemat(ecg_save_path + '/%s.mat'%infor[i].ecg_id, {'ecgraw' : input[i]})

        # a += 1
        #
        # if a > 20:
        #     break
    print(len(train_loader.dataset))
    print(a)

def run_check_DataSet():
    val_dataset = CinCDataset(
        mode='train',
        csv='train.csv',
        split='valid_a%d_4302.npy' % (0),
    )

    # label_save_path = 'F:/data_root/CinC2020/check_label'
    # ecg_save_path = DATA_DIR + '/check_ecg'

    a = 0
    for t, (input, truth, infor) in enumerate(val_dataset):

        # sio.savemat(ecg_save_path + '/%s.mat'%infor.ecg_id, {'ecgraw' : input})

        a += 1

        if a > 20:
            break

    print(len(val_dataset))
    print(a)


# main #################################################################
if __name__ == '__main__':
    run_check_DataSet()

    print('\nsucess!')