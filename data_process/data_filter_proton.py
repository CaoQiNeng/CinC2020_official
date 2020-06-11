from scipy.signal import butter, lfilter
import scipy.io as sio
from include import *

def bandpass_filter(data, lowcut, highcut, signal_freq, filter_order):
    """
    Method responsible for creating and applying Butterworth filter.
    :param deque data: raw data
    :param float lowcut: filter lowcut frequency value
    :param float highcut: filter highcut frequency value
    :param int signal_freq: signal frequency in samples per second (Hz)
    :param int filter_order: filter order
    :return array: filtered data
    """
    nyquist_freq = 0.5 * signal_freq
    low = lowcut / nyquist_freq
    high = highcut / nyquist_freq
    b, a = butter(filter_order, [low, high], btype="band")
    y = lfilter(b, a, data)
    return y

# target = 'A5404'
# temp_ecg = sio.loadmat(DATA_ROOT_PATH + '/CinC2020/Training_WFDB/' + target + '.mat')['val']
# import json
# os.makedirs(DATA_ROOT_PATH + '/CinC2020/Training_WFDB_json', exist_ok=True)
# with open(DATA_ROOT_PATH + '/CinC2020/Training_WFDB_json/' + target + '.json','w') as f:
#     str = json.dumps({'ecgraw': (temp_ecg[0, :] / 1000).tolist(), 'sampr': 500})
#     f.write(str)

data_path = DATA_ROOT_PATH + '/CinC2020/Training_WFDB'
data_list = glob.glob(DATA_ROOT_PATH + '/CinC2020/Training_WFDB/*.mat')
os.makedirs(data_path + '_3fil', exist_ok=True)

for i, d in enumerate(data_list):
    fn = (os.path.split(d)[-1]).split('.')[0]
    print(fn)
    data = sio.loadmat(DATA_ROOT_PATH + '/CinC2020/Training_WFDB/%s.mat' % fn)
    temp_ecg = data['val']
    temp_ecg = temp_ecg / 1000
    for i in range(temp_ecg.shape[0]):
        temp_ecg[i, :] = bandpass_filter(temp_ecg[i, :], lowcut=0.5, highcut=49.0,
                                         signal_freq=500, filter_order=1)

        temp_ecg[i, :] = bandpass_filter(temp_ecg[i, :], lowcut=0.5, highcut=49.0,
                                         signal_freq=500, filter_order=1)

        temp_ecg[i, :] = bandpass_filter(temp_ecg[i, :], lowcut=0.5, highcut=49.0,
                                         signal_freq=500, filter_order=1)

    sio.savemat(data_path + '_3fil/' + fn + '.mat',{'val':temp_ecg})



# os.makedirs(DATA_ROOT_PATH + '/CinC2020/Training_WFDB_fil', exist_ok=True)
# sio.savemat(DATA_ROOT_PATH + '/CinC2020/Training_WFDB_fil/'+ target + '_2.mat' ,{'ecgfil': temp_ecg})
