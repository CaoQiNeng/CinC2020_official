from include import *

def do_label_stat(df):
    # ['AF', 'I-AVB', 'LBBB', 'Normal', 'PAC', 'PVC', 'RBBB', 'STD', 'STE']
    class_map = [164889003, 270492004, 164909002, 426783006, 284470004,  164884008, 59118001, 429622005, 164931005]
    class_map_num = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    df = df.replace(math.nan, 0, regex=True)

    for i in range(len(df)) :
        d = df.iloc[i]
        for j in range(len(d)) :
            if d[j] in class_map:
                d_index = class_map.index(d[j])
                class_map_num[d_index] = class_map_num[d_index] + 1
        # d1 = d['First_label']
        # d2 = d['Second_label']
        # d3 = d['Third_label']
        #
        # d1_index = class_map.index(d1)
        # class_map_num[d1_index] = class_map_num[d1_index] + 1
        # if not d2 == 0 :
        #     d2_index = class_map.index(d2)
        #     class_map_num[d2_index] = class_map_num[d2_index] + 1
        # if not d3 == 0 :
        #     d3_index = class_map.index(d3)
        #     class_map_num[d3_index] = class_map_num[d3_index] + 1


    return class_map_num

# ['AF', 'I-AVB', 'LBBB', 'Normal', 'PAC', 'PVC', 'RBBB', 'STD', 'STE']
stat_df = pd.DataFrame(columns = [164889003, 270492004, 164909002, 426783006, 284470004, 164884008, 59118001, 429622005, 164931005])
target = 'SNOMED_PhysioNetChallenge2020_PTB-XL_label'
csv_path = DATA_ROOT_PATH + '/CinC2020_V1/' + target + '.csv'
overall_df = pd.read_csv(csv_path)
overall_label_stat = do_label_stat(overall_df)

stat_df.loc['overall'] = overall_label_stat

# overall_df = overall_df.set_index('Recording')
# for i in range(10):
#     np_dat = np.load('F:/data_root/CinC2020/split_v2/test-a%d_0-687.npy'%i)
#     df = overall_df.loc[np_dat]
#
#     label_stat = do_label_stat(df)
#     stat_df.loc['test-a%d_1-687.npy'%i] = label_stat

stat_df.to_csv(DATA_ROOT_PATH + '/CinC2020_V1/Label_Distri_' + target + '.csv')

