from common import *

#--------------
DATA_DIR = DATA_ROOT_PATH + '/CinC2020'
class_map = ['AF', 'I-AVB', 'LBBB', 'Normal', 'PAC',  'PVC', 'RBBB', 'STD', 'STE']

class CinCDataset(Dataset):
    def __init__(self, split, mode, csv):
        def seg_data_to_5s(ecg):
            ecg_5s = []
            block_len = 5 * 500
            ecg_len = ecg.shape[1]
            if ecg_len % block_len == 0 :
                block_num = int(ecg_len / block_len)
            else:
                block_num = int(ecg_len / block_len) + 1

            for i in range(block_num):
                if i == block_num - 1 :
                    ecg_5s.append(ecg[:, -block_len : ])
                else:
                    ecg_5s.append(ecg[:, i * block_len : (i + 1) * block_len])


            ecg_5s = np.stack(ecg_5s)

            return ecg_5s


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
        self.First_label = df['First_label'].values
        self.Second_label = df['Second_label'].values
        self.Third_label = df['Third_label'].values

        self.First_label[self.First_label == class_map[0]] = class_map.index(class_map[0])
        self.Second_label[self.Second_label == class_map[0]] = class_map.index(class_map[0])
        self.Third_label[self.Third_label == class_map[0]] = class_map.index(class_map[0])

        for c in class_map[1:] :
            self.First_label[self.First_label == c] = class_map.index(class_map[3])
            self.Second_label[self.Second_label == c] = class_map.index(class_map[3])
            self.Third_label[self.Third_label == c] = class_map.index(class_map[3])

        self.Recording_5s = []
        self.ecg_5s = []
        self.First_label_5s = []
        self.Second_label_5s = []
        self.Third_label_5s = []
        for i, r in enumerate(self.Recording):
            ecg = sio.loadmat(DATA_DIR + '/Training_WFDB/%s.mat' % r)['val']
            temp_ecg = seg_data_to_5s(ecg)

            block_num = len(temp_ecg)
            for j in range(block_num):
                self.Recording_5s.append('%s_%02d' % (r, j))
                self.First_label_5s.append(self.First_label[i])
                self.Second_label_5s.append(self.Second_label[i])
                self.Third_label_5s.append(self.Third_label[i])

            self.ecg_5s.append(temp_ecg)

        self.Recording_5s = np.vstack(self.Recording_5s)
        self.ecg_5s = np.vstack(self.ecg_5s)
        self.First_label_5s = np.vstack(self.First_label_5s)
        self.Second_label_5s = np.vstack(self.Second_label_5s)
        self.Third_label_5s = np.vstack(self.Third_label_5s)

        self.num_image = len(self.Recording_5s)

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
        ecg_id = self.Recording_5s[index]
        First_label = self.First_label_5s[index]
        Second_label = self.Second_label_5s[index]
        Third_label = self.Third_label_5s[index]

        label = np.zeros(len(class_map))
        label[First_label] = 1
        if not math.isnan(Second_label):
            label[int(Second_label)] = 1
        if not math.isnan(Third_label):
            label[int(Third_label)] = 1

        label[1:] = 0

        ecg = self.ecg_5s[index]
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
    train_fold = 0
    valid_fold = 0
    batch_size = 72
    train_dataset = CinCDataset(
        mode='train',
        csv='train.csv',
        split='train-a%d_%d-5571.npy' % (train_fold, valid_fold),
    )
    train_loader = DataLoader(
        train_dataset,
        # sampler     = RandomSampler(train_dataset),
        shuffle=True,
        batch_size=batch_size,
        drop_last=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=null_collate
    )

    label_save_path = 'F:/data_root/CinC2020/check_label'
    ecg_save_path = 'F:/data_root/CinC2020/check_ecg'

    a = 0
    for t, (input, truth, infor) in enumerate(train_dataset):
        print(truth)
        # a += len(infor)
        # print(len(infor))
        print(infor.ecg_id)
        # for i in range(len(infor)):
        with open(label_save_path + '/' + infor.ecg_id[0] + '.json', 'w') as f:
            json_str = json.dumps({'label' : truth.tolist()})
            f.write(json_str)

        sio.savemat(ecg_save_path + '/%s.mat'%infor.ecg_id[0], {'ecgraw' : input})

        # a += 1
        #
        # if a > 20:
        #     break
    print(len(train_loader.dataset))
    print(a)


# main #################################################################
if __name__ == '__main__':
    run_check_DataLoader()

    print('\nsucess!')