from common import *

#--------------
DATA_DIR = DATA_ROOT_PATH + '/CinC2020'
class_map = ['AF', 'I-AVB', 'LBBB', 'Normal', 'PAC',  'PVC', 'RBBB', 'STD', 'STE']

class CinCDataset(Dataset):
    def __init__(self, split, mode, csv, data_path):
        self.split   = split
        self.csv     = csv
        self.mode    = mode
        self.data_path = data_path

        self.data_list = np.array(os.listdir(data_path))
        self.data_list = np.char.replace(self.data_list, '.mat', '')

        df = pd.read_csv(DATA_DIR + '/cinc2020_label.csv') #.fillna('')

        self.Recording = df['Recording'].values
        self.First_label = df['First_label'].values
        self.Second_label = df['Second_label'].values
        self.Third_label = df['Third_label'].values

        self.First_label[self.First_label == class_map[0]] = class_map.index(class_map[0])
        self.Second_label[self.Second_label == class_map[0]] = class_map.index(class_map[0])
        self.Third_label[self.Third_label == class_map[0]] = class_map.index(class_map[0])

        for c in class_map[1:]:
            self.First_label[self.First_label == c] = class_map.index(class_map[3])
            self.Second_label[self.Second_label == c] = class_map.index(class_map[3])
            self.Third_label[self.Third_label == c] = class_map.index(class_map[3])

        self.num_image = len(self.data_list)

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
        def remove_outlier(ecg):
            if (ecg > 3).any():
                for i in range(12):
                    b = np.argwhere(np.diff(ecg[i]) > 3)
                    if b.shape[0] > 0:
                        for k in b[:, 0]:
                            ecg[i][k + 1] = ecg[i][k]
            if (ecg < -3).any():
                for i in range(12):
                    b = np.argwhere(np.diff(ecg[i]) < -3)
                    if b.shape[0] > 0:
                        for k in b[:, 0]:
                            ecg[i][k + 1] = ecg[i][k]
            return ecg

        ecg_id = self.data_list[index]
        label_index = np.where(self.Recording == ecg_id[:-3])[0][0]
        First_label = self.First_label[label_index]
        Second_label = self.Second_label[label_index]
        Third_label = self.Third_label[label_index]

        label = np.zeros(len(class_map))
        label[First_label] = 1
        if not math.isnan(Second_label):
            label[Second_label] = 1
        if not math.isnan(Third_label):
            label[Third_label] = 1

        label[1:] = 0

        ecg = sio.loadmat(self.data_path + '/%s.mat'%ecg_id)['ecgfil']
        ecg = np.array(ecg, dtype=np.float32)
        ecg = np.nan_to_num(ecg)
        ecg = remove_outlier(ecg)

        infor = Struct(
            index  = index,
            ecg_id = ecg_id,
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

def run_check_Dataset():
    batch_size = 72
    train_dataset = CinCDataset(
        mode='train',
        csv='train.csv',
        split='valid_a%d_687.npy' % 0,
        data_path = DATA_DIR + '/data_argument/10s/train_data'
    )

    val_dataset = CinCDataset(
        mode='train',
        csv='train.csv',
        split='valid_a%d_687.npy' % 0,
        data_path=DATA_DIR + '/data_argument/5s_a3_0/valid_data'
    )

    label_save_path = 'F:/data_root/CinC2020/check_label'
    ecg_save_path = 'F:/data_root/CinC2020/check_ecg'

    a = 0
    for t, (input, truth, infor) in enumerate(val_dataset):
        if input.shape[1] != 5000:
            print(infor.ecg_id)
            print(input.shape[1])
            # print(truth)
            exit()

        # if t ==  10 :
        #     break
    print(len(train_dataset))
    print(a)

def run_check_DataLoader():
    batch_size = 72
    train_dataset = CinCDataset(
        mode='train',
        csv='train.csv',
        split='valid_a%d_687.npy' % 0,
        data_path = DATA_DIR + '/data_argument/10s/train_data'
    )

    val_dataset = CinCDataset(
        mode='train',
        csv='train.csv',
        split='valid_a%d_687.npy' % 0,
        data_path=DATA_DIR + '/data_argument/10s/valid_data'
    )

    train_loader = DataLoader(
        train_dataset,
        # sampler     = RandomSampler(train_dataset),
        shuffle=True,
        batch_size=batch_size,
        drop_last=False,
        num_workers=20,
        pin_memory=True,
        collate_fn=null_collate
    )

    label_save_path = 'F:/data_root/CinC2020/check_label'
    ecg_save_path = 'F:/data_root/CinC2020/check_ecg'

    a = 0
    for t, (input, truth, infor) in enumerate(train_loader):
        print(infor.ecg_id)
        print(truth)

        # if t ==  10 :
        #     break
    print(len(train_dataset))
    print(a)


# main #################################################################
if __name__ == '__main__':
    # run_check_DataLoader()
    run_check_Dataset()

    print('\nsucess!')