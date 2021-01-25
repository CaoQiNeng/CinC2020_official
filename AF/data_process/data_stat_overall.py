import pandas as pd
import math
import numpy as np

def do_label_stat(df):
    class_map = ['AF', 'I-AVB', 'LBBB', 'Normal', 'PAC', 'PVC', 'RBBB', 'STD', 'STE']
    class_map_num = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    df = df.replace(math.nan, 0, regex=True)

    for i in range(len(df)) :
        d = df.iloc[i]
        print(d)
        d1 = d['First_label']
        d2 = d['Second_label']
        d3 = d['Third_label']

        d1_index = class_map.index(d1)
        class_map_num[d1_index] = class_map_num[d1_index] + 1
        if not d2 == 0 :
            d2_index = class_map.index(d2)
            class_map_num[d2_index] = class_map_num[d2_index] + 1
        if not d3 == 0 :
            d3_index = class_map.index(d3)
            class_map_num[d3_index] = class_map_num[d3_index] + 1

    return class_map_num

stat_df = pd.DataFrame(columns = ['AF', 'I-AVB', 'LBBB', 'Normal', 'PAC', 'PVC', 'RBBB', 'STD', 'STE'])
csv_path = 'F:/data_root/CinC2020/cinc2020_label.csv'
overall_df = pd.read_csv(csv_path)
overall_label_stat = do_label_stat(overall_df)

stat_df.loc['overall'] = overall_label_stat

overall_df = overall_df.set_index('Recording')
for i in range(10):
    np_dat = np.load('F:/data_root/CinC2020/split_v2/test-a%d_0-687.npy'%i)
    df = overall_df.loc[np_dat]

    label_stat = do_label_stat(df)
    stat_df.loc['test-a%d_1-687.npy'%i] = label_stat

stat_df.to_csv('test_label_stat.csv')

