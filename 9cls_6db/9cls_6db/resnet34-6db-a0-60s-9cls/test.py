from scipy import io as sio
import glob

# files = sorted(glob.glob('F:/data_root/CinC2020/Training_WFDB/*.mat'))
#
# max_len = 0
# for f in files:
#     data = sio.loadmat(f)['val']
#     data_len = data.shape[1]
#     if data_len > max_len:
#         max_len = data_len
#         print(max_len)

print(72000 / 256 / 5)