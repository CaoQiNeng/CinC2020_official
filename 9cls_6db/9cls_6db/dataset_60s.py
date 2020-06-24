from common import *
from scipy import signal

#--------------
DATA_DIR = DATA_ROOT_PATH + '/CinC2020'
class_map = ['AF', 'I-AVB', 'LBBB', 'Normal', 'PAC',  'PVC', 'RBBB', 'STD', 'STE']

class CinCDataset(Dataset):
    def __init__(self, split, mode, csv):
        def pick_label(overall_labels):
            picked_labels = [math.nan, math.nan, math.nan]
            for i in range(len(overall_labels)):
                if overall_labels[i] in class_map :
                    picked_labels[i] = overall_labels[i]

            return picked_labels

        self.split   = split
        self.csv     = csv
        self.mode    = mode

        df = pd.read_csv(DATA_DIR + '/cinc2020_label.csv') #.fillna('')

        if split is not None:
            all_data = df['Recording'].values
            s = np.load(DATA_DIR + '/split_v2/%s' % split)
            s_data = []
            for d in all_data:
                if d[:] in s:
                    s_data.append(d)

            df = df_loc_by_list(df, 'Recording', s_data)

        self.Recording = df['Recording'].values

        # self.First_label = df['First_label'].values
        # self.Second_label = df['Second_label'].values
        # self.Third_label = df['Third_label'].values
        #
        # for c in class_map:
        #     self.First_label[self.First_label == c] = class_map.index(c)
        #     self.Second_label[self.Second_label == c] = class_map.index(c)
        #     self.Third_label[self.Third_label == c] = class_map.index(c)

        self.num_image = len(self.Recording)

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
        First_label = self.First_label[index]
        Second_label = self.Second_label[index]
        Third_label = self.Third_label[index]

        label = np.zeros(len(class_map))
        label[First_label] = 1
        if not math.isnan(Second_label):
            label[Second_label] = 1
        if not math.isnan(Third_label):
            label[Third_label] = 1

        temp_ecg = sio.loadmat(DATA_DIR + '/Training_WFDB/%s.mat' % ecg_id)['val']
        if ecg_id.startswith('S'):
            temp_ecg = resample(temp_ecg, 1000)
        else:
            temp_ecg = resample(temp_ecg, 500)

        ecg = np.zeros((12, 18000), dtype=np.float32)
        ecg[:,-temp_ecg.shape[1]:] = temp_ecg[:,-18000:]
        ecg = np.array(ecg / 1000, dtype=np.float32)

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
        print(truth)
        exit()
        a += len(infor)
        print(len(infor))
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
        split='valid-a%d-4302.npy' % (0),
    )

    label_save_path = 'F:/data_root/CinC2020/check_label'
    ecg_save_path = 'F:/data_root/CinC2020/check_ecg'

    a = 0
    for t, (input, truth, infor) in enumerate(val_dataset):
        print(truth)
        exit()
        a += len(infor)
        print(len(infor))
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

    print(len(val_dataset))
    print(a)


# main #################################################################
if __name__ == '__main__':
    run_check_DataSet()

    print('\nsucess!')