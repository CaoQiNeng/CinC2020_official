import os
import pandas as pd
from include import *

target = 'PhysioNetChallenge2020_PTB-XL'
root_path = DATA_ROOT_PATH + '/CinC2020_V1'
st_df = pd.read_csv(root_path + '/SNOMED-CT.csv')
st_df = st_df.set_index('SNOMED_code')

hea_path = root_path + '/' + target
Recording = []
First_label = []
Second_label = []
Third_label = []
Fourth_label = []
Fifth_label = []
Sixth_label = []
Seventh_label = []
Eighth_label = []
Ninth_label = []
Tenth_label = []
Eleventh_label = []
data_len = []
len_l = 0
for f in os.listdir(hea_path):
    if f.lower().endswith('hea'):
        file = open(hea_path + "/" + f)
        print(f)
        for line in file:
            if line.startswith("#Dx:"):
                Recording.append(f.replace(".hea",""))
                line = line.replace('#Dx: ','')
                line = line.replace('\n','')
                labels = line.split(',')
                # l = st_df.loc[int(labels[0])]['Abbreviation']
                First_label.append(int(labels[0]))
                if len(labels) > 1:
                    # l = st_df.loc[int(labels[1])]['Abbreviation']
                    Second_label.append(int(labels[1]))
                else:
                    Second_label.append('')

                if len(labels) > 2:
                    # l = st_df.loc[int(labels[2])]['Abbreviation']
                    Third_label.append(int(labels[2]))
                else:
                    Third_label.append('')

                if len(labels) > 3:
                    # l = st_df.loc[int(labels[3])]['Abbreviation']
                    Fourth_label.append(int(labels[3]))
                else:
                    Fourth_label.append('')

                if len(labels) > 4:
                    # l = st_df.loc[int(labels[4])]['Abbreviation']
                    Fifth_label.append(int(labels[4]))
                else:
                    Fifth_label.append('')

                if len(labels) > 5:
                    # l = st_df.loc[int(labels[5])]['Abbreviation']
                    Sixth_label.append(int(labels[5]))
                else:
                    Sixth_label.append('')

                if len(labels) > 6:
                    # l = st_df.loc[int(labels[6])]['Abbreviation']
                    Seventh_label.append(int(labels[6]))
                else:
                    Seventh_label.append('')

                if len(labels) > 7:
                    # l = st_df.loc[int(labels[7])]['Abbreviation']
                    Eighth_label.append(int(labels[7]))
                else:
                    Eighth_label.append('')

                if len(labels) > 8:
                    # l = st_df.loc[int(labels[8])]['Abbreviation']
                    Ninth_label.append(int(labels[8]))
                else:
                    Ninth_label.append('')

                if len(labels) > 9:
                    # l = st_df.loc[int(labels[9])]['Abbreviation']
                    Tenth_label.append(int(labels[9]))
                else:
                    Tenth_label.append('')

                if len(labels) > 10:
                    # l = st_df.loc[int(labels[10])]['Abbreviation']
                    Eleventh_label.append(int(labels[10]))
                else:
                    Eleventh_label.append('')

                if len(labels) > len_l :
                    len_l = len(labels)

        data = sio.loadmat(root_path + '/' + target + '/'  + "/" + f.replace('hea', 'mat'))['val']
        data_len.append(data.shape[1])


df = pd.DataFrame(zip(Recording, First_label, Second_label, Third_label, Fourth_label, Fifth_label, Sixth_label,
                      Seventh_label, Eighth_label, Ninth_label, Tenth_label, Eleventh_label, data_len),
                  columns=['Recording', 'First_label', 'Second_label', 'Third_label', 'Fourth_label',
                           'Fifth_label', 'Sixth_label', 'Seventh_label', 'Eighth_label', 'Ninth_label', 'Tenth_label', 'Eleventh_label', 'data_len'])

df = df.sort_values(by = 'Recording')
df.to_csv(root_path + '/' + target + '_label_SNOMED.csv', index=False)

print(len_l)