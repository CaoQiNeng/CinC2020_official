from include import *
import scipy.io as sio

DATA_DIR = DATA_ROOT_PATH + '/CinC2020/Training_WFDB_3fil'
save_path = DATA_ROOT_PATH + '/CinC2020/data_argument'
os.makedirs(save_path, exist_ok=True)
save_path_5s = save_path + '/5s_3fil_a3_0'
save_path_10s = save_path + '/10s_3fil_a3_0'
os.makedirs(save_path_5s, exist_ok=True)

def seg_data_to_5s():
    train_data_ids = np.load(DATA_ROOT_PATH + '/CinC2020/split_v2/train-a3_0-5571.npy')
    os.makedirs(save_path_5s + '/train_data', exist_ok=True)
    valid_data_ids = np.load(DATA_ROOT_PATH + '/CinC2020/split_v2/valid-a3_0-619.npy')
    os.makedirs(save_path_5s + '/valid_data', exist_ok=True)
    test_data_ids = np.load(DATA_ROOT_PATH +  '/CinC2020/split_v2/test-a3_0-687.npy')
    os.makedirs(save_path_5s + '/test_data', exist_ok=True)

    for i, t in enumerate(train_data_ids):
        data = sio.loadmat(DATA_DIR + '/' + t)['val']
        data_len = data.shape[1]
        block_len = 5 * 500
        if data_len % block_len == 0:
            block_num = int(data_len / block_len)
        else:
            block_num = int(data_len / block_len) + 1

        for j in range(block_num):
            if j == block_num - 1:
                sio.savemat(save_path_5s + '/train_data/%s_%02d.mat'%(t, j),{'ecgfil':data[:, -block_len:]})
            else:
                sio.savemat(save_path_5s + '/train_data/%s_%02d.mat'%(t, j),{'ecgfil':data[:, j * block_len: (j + 1) * block_len]})

    for i, t in enumerate(test_data_ids):
        data = sio.loadmat(DATA_DIR + '/' + t)['val']
        data_len = data.shape[1]
        block_len = 5 * 500
        if data_len % block_len == 0:
            block_num = int(data_len / block_len)
        else:
            block_num = int(data_len / block_len) + 1

        for j in range(block_num):
            if j == block_num - 1:
                sio.savemat(save_path_5s + '/test_data/%s_%02d.mat'%(t, j),{'ecgfil':data[:, -block_len:]})
            else:
                sio.savemat(save_path_5s + '/test_data/%s_%02d.mat'%(t, j),{'ecgfil':data[:, j * block_len: (j + 1) * block_len]})

    for i, t in enumerate(valid_data_ids):
        data = sio.loadmat(DATA_DIR + '/' + t)['val']
        data_len = data.shape[1]
        block_len = 5 * 500
        if data_len % block_len == 0:
            block_num = int(data_len / block_len)
        else:
            block_num = int(data_len / block_len) + 1

        for j in range(block_num):
            if j == block_num - 1:
                sio.savemat(save_path_5s + '/valid_data/%s_%02d.mat'%(t, j),{'ecgfil':data[:, -block_len:]})
            else:
                sio.savemat(save_path_5s + '/valid_data/%s_%02d.mat'%(t, j),{'ecgfil':data[:, j * block_len: (j + 1) * block_len]})

def seg_data_to_10s():
    train_data_ids = np.load(DATA_ROOT_PATH + '/CinC2020/split_v2/train-a3_0-5571.npy')
    os.makedirs(save_path_10s + '/train_data', exist_ok=True)
    valid_data_ids = np.load(DATA_ROOT_PATH + '/CinC2020/split_v2/valid-a3_0-619.npy')
    os.makedirs(save_path_10s + '/valid_data', exist_ok=True)
    test_data_ids = np.load(DATA_ROOT_PATH + '/CinC2020/split_v2/test-a3_0-687.npy')
    os.makedirs(save_path_10s + '/test_data', exist_ok=True)
    for i, t in enumerate(train_data_ids):
        data = sio.loadmat(DATA_DIR + '/' + t)['val']
        data_len = data.shape[1]
        block_len = 10 * 500
        if data_len % block_len != 0 and data_len % block_len >= 4 * 500:
            block_num = int(data_len / block_len) + 1
        else:
            block_num = int(data_len / block_len)

        if data_len < block_len:
            temp_ecg = np.zeros((12, block_len), dtype=np.float32)
            temp_ecg[:, -data_len:] = data[:, -block_len:]
            sio.savemat(save_path_10s + '/train_data/%s_%02d.mat' % (t, 0), {'ecgfil': temp_ecg})
            continue

        for j in range(block_num):
            if j == block_num - 1:
                sio.savemat(save_path_10s + '/train_data/%s_%02d.mat'%(t, j),{'ecgfil':data[:, -block_len:]})
            else:
                sio.savemat(save_path_10s + '/train_data/%s_%02d.mat'%(t, j),{'ecgfil':data[:, j * block_len: (j + 1) * block_len]})

    for i, t in enumerate(valid_data_ids):
        data = sio.loadmat(DATA_DIR + '/' + t)['val']
        data_len = data.shape[1]
        block_len = 10 * 500
        if data_len % block_len != 0 and data_len % block_len >= 4 * 500:
            block_num = int(data_len / block_len) + 1
        else:
            block_num = int(data_len / block_len)

        if data_len < block_len:
            temp_ecg = np.zeros((12, block_len), dtype=np.float32)
            temp_ecg[:, -data_len:] = data[:, -block_len:]
            sio.savemat(save_path_10s + '/valid_data/%s_%02d.mat' % (t, 0), {'ecgfil': temp_ecg})
            continue

        for j in range(block_num):
            if j == block_num - 1:
                sio.savemat(save_path_10s + '/valid_data/%s_%02d.mat' % (t, j), {'ecgfil': data[:, -block_len:]})
            else:
                sio.savemat(save_path_10s + '/valid_data/%s_%02d.mat' % (t, j),
                            {'ecgraw': data[:, j * block_len: (j + 1) * block_len]})

    for i, t in enumerate(test_data_ids):
        data = sio.loadmat(DATA_DIR + '/' + t)['val']
        data_len = data.shape[1]
        block_len = 10 * 500
        if data_len % block_len != 0 and data_len % block_len >= 4 * 500:
            block_num = int(data_len / block_len) + 1
        else:
            block_num = int(data_len / block_len)

        if data_len < block_len:
            temp_ecg = np.zeros((12, block_len), dtype=np.float32)
            temp_ecg[:, -data_len:] = data[:, -block_len:]
            sio.savemat(save_path_10s + '/test_data/%s_%02d.mat' % (t, 0), {'ecgfil': temp_ecg})
            continue

        for j in range(block_num):
            if j == block_num - 1:
                sio.savemat(save_path_10s + '/test_data/%s_%02d.mat' % (t, j), {'ecgfil': data[:, -block_len:]})
            else:
                sio.savemat(save_path_10s + '/test_data/%s_%02d.mat' % (t, j),
                            {'ecgfil': data[:, j * block_len: (j + 1) * block_len]})


seg_data_to_5s()