import os
import pandas as pd
from include import *

def not_in_fulllist():
    target = 'PhysioNetChallenge2020_Training_E'
    root_path = DATA_ROOT_PATH + '/CinC2020_V1'
    st_df = pd.read_csv(root_path + '/SNOMED-CT.csv')
    st_df = st_df['SNOMED_code']
    st_df = st_df.tolist()

    hea_path = root_path + '/' + target
    f_list = []
    for f in os.listdir(hea_path):
        if f.lower().endswith('hea'):
            file = open(hea_path + "/" + f)
            print(f)
            for line in file:
                if line.startswith("#Dx:"):
                    line = line.replace('#Dx: ','')
                    line = line.replace('\n','')
                    labels = line.split(',')
                    # l = st_df.loc[int(labels[0])]['Abbreviation']

                    for l in labels:
                        if int(l) not in st_df:
                            f_list.append(f.replace('.hea',''))

    df = pd.DataFrame(zip(f_list), columns=['f_list'])

    df.to_csv(root_path + '/' + target + '_data_not_in_fulllist.csv', index=False)


def not_in_unscored():
    target = 'PhysioNetChallenge2020_Training_StPetersburg'
    root_path = DATA_ROOT_PATH + '/CinC2020_V1'
    temp_scored_map = pd.read_csv(root_path + '/evaluation-2020-master/dx_mapping_scored.csv')['SNOMED CT Code'].tolist()
    temp_scored_map = [str(i) for i in temp_scored_map]

    hea_path = root_path + '/' + target
    f_list = []
    for f in os.listdir(hea_path):
        if f.lower().endswith('hea'):
            file = open(hea_path + "/" + f)
            print(f)
            for line in file:
                if line.startswith("#Dx:"):
                    line = line.replace('#Dx: ', '')
                    line = line.replace('\n', '')
                    labels = line.split(',')

                    # l = st_df.loc[int(labels[0])]['Abbreviation']

                    if len(set(labels) & set(temp_scored_map)) == 0 :
                        f_list.append(f.replace('.hea', ''))

    df = pd.DataFrame(zip(f_list), columns=['f_list'])

    df.to_csv(root_path + '/' + target + '_data_not_in_scored.csv', index=False)


not_in_unscored()