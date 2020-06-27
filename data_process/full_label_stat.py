import os
import pandas as pd
from include import *

target = 'overall_hea'
root_path = DATA_ROOT_PATH + '/CinC2020_V1'
st_df = pd.read_csv(root_path + '/SNOMED-CT.csv')
st_df = st_df['SNOMED_code']
st = np.array(st_df.tolist())

st_num = np.zeros_like(st)


hea_path = root_path + '/' + target
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

                label_dict = {}
                for l in labels:
                    index = np.where(st == int(l))[0]
                    if len(index) != 0:
                        st_num[index[0]] = st_num[index[0]] + 1


df = pd.DataFrame(zip(st, st_num), columns=['st', 'st_num'])

df.to_csv(root_path + '/' + target + '_full_label_stat.csv', index=False)