from common import *
from scipy import signal
from scipy.signal import butter, lfilter

DATA_DIR = DATA_ROOT_PATH + '/CinC2020_V1'
df = pd.read_csv(DATA_DIR + '/SNOMED_overall_label.csv') #.fillna('')

Recording = df['Recording'].values
df = df.set_index('Recording')

def resample(data, sampr, after_Hz=300):
    data_len = len(data)
    propessed_data = signal.resample(data, int(data_len * (after_Hz / sampr)))

    return propessed_data

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

for i in range(len(df)) :
    ecg_id = Recording[i]
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

    for i in range(temp_ecg.shape[0]):
        temp_ecg[i, :] = bandpass_filter(temp_ecg[i, :], lowcut=0.5, highcut=49.0,
                                         signal_freq=500, filter_order=1)

        temp_ecg[i, :] = bandpass_filter(temp_ecg[i, :], lowcut=0.5, highcut=49.0,
                                         signal_freq=500, filter_order=1)

        temp_ecg[i, :] = bandpass_filter(temp_ecg[i, :], lowcut=0.5, highcut=49.0,
                                         signal_freq=500, filter_order=1)

    sio.savemat(DATA_DIR + '/overall_3fil/' + ecg_id + '.mat', {'val': temp_ecg})



