import os
import pandas as pd
from include import *

target = 'PhysioNetChallenge2020_Training_StPetersburg'
root_path = DATA_ROOT_PATH + '/CinC2020_V1'
st_df = pd.read_csv(root_path + '/SNOMED-CT.csv')
st_df = st_df['SNOMED_code']
st_df = st_df.tolist()

hea_path = root_path + '/' + target
sc = []
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
                    if int(l) not in st_df and int(l) not in sc :
                        sc.append(int(l))

df = pd.DataFrame(zip(sc), columns=['sc'])

df.to_csv(root_path + '/' + target + '_not_in_fulllist.csv', index=False)
