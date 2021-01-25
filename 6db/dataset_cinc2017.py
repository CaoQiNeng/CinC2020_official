from common import *
from scipy import signal

#--------------
DATA_DIR = DATA_ROOT_PATH + '/CinC2017'
# ['AF', 'I-AVB', 'LBBB', 'Normal', 'PAC',  'PVC', 'RBBB', 'STD', 'STE']
class_map = np.array(['A', 'N', 'O', '~'])

class CinCDataset(Dataset):
    def __init__(self, split):
        self.split = split
        temp_df = pd.read_csv(DATA_DIR + '/training2017/REFERENCE_V2.csv') #.fillna('')
        s = np.load(DATA_DIR + '/training2017/' + split)

        self.featrue_list = sio.loadmat(DATA_DIR + '/feature_st.mat')['features_st']

        df = pd.DataFrame(columns = ['ids', 'labels'])
        q = 0
        for i in range(len(temp_df)):
            if temp_df['ids'].iloc[i] in s:
                df.loc[q] = temp_df.iloc[i].values

                q += 1

        self.ids = df['ids'].values
        self.labels = []
        df = df.set_index('ids')

        for i in range(len(self.ids)) :
            d = df.loc[self.ids[i]]

            label = np.zeros(4)
            l_index = np.where(class_map == d[0])
            if len(l_index[0]) != 0:
                label[l_index[0][0]] = 1

            self.labels.append(label)

        self.num_image = len(df)

    def __str__(self):
        string  = ''
        string += '\tlen = %d\n'%len(self)
        string += '\n'
        string += '\tnum_image = %d\n'%self.num_image
        string += '\n'
        string += '\tsplit = %s\n' % self.split
        return string

    def __len__(self):
        return self.num_image

    def __getitem__(self, index):
        ecg_id = self.ids[index]

        label = self.labels[index]

        temp_ecg = sio.loadmat(DATA_DIR + '/training2017/%s.mat' % ecg_id)['val']
        temp_ecg = np.array(temp_ecg / 1000)

        ecg = np.zeros((1, 9000), dtype=np.float32)
        ecg[:,-temp_ecg.shape[1]:] = temp_ecg[:,-9000:]

        infor = Struct(
            index  = index,
            ecg_id = ecg_id,
        )

        return ecg, label, infor, self.featrue_list[index]

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
    f = []
    for b in range(batch_size):
        input.append(batch[b][0])
        label.append(batch[b][1])
        infor.append(batch[b][2])
        f.append(batch[b][3])

    label = torch.from_numpy(np.stack(label)).float()
    input = torch.from_numpy(np.stack(input)).float()
    f = torch.from_numpy(np.stack(f)).float()

    return input, label, infor, f

def run_check_DataLoader():
    batch_size = 72
    train_dataset = CinCDataset('train_a2_7676.npy')
    train_loader = DataLoader(
        train_dataset,
        sampler=RandomSampler(train_dataset),
        # sampler = CustomSampler(train_dataset),
        # shuffle=True,
        batch_size=batch_size,
        drop_last=False,
        num_workers=1,
        pin_memory=True,
        collate_fn=null_collate
    )

    label_save_path = 'F:/data_root/CinC2020/check_label'
    ecg_save_path = 'F:/data_root/CinC2020/check_ecg'

    a = 0
    for t, (input, truth, infor, f) in enumerate(train_loader):
        print(infor[0].ecg_id)
        print(truth)
        print(f)
        exit()
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
        split='train_a0_7676.npy',
    )

    # label_save_path = 'F:/data_root/CinC2020/check_label'
    ecg_save_path = DATA_DIR + '/check_ecg'

    a = 0
    for t, (input, truth, infor, f) in enumerate(val_dataset):
        print(infor.ecg_id)
        print(truth)
        print(f)
        exit()
        # sio.savemat(ecg_save_path + '/%s.mat'%infor.ecg_id, {'ecgraw' : input})

        a += 1

        if a > 20:
            break

    print(len(val_dataset))
    print(a)


# main #################################################################
if __name__ == '__main__':
    run_check_DataLoader()

    print('\nsucess!')