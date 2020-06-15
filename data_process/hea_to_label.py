import os
import pandas as pd

target = 'PhysioNetChallenge2020_Training_2'
root_path = '/home1/cqn/data_root/CinC2020_V1'
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
len_l = 0
for f in os.listdir(hea_path):
    if f.lower().endswith('hea'):
        file = open(hea_path + "/" + f)
        for line in file:
            if line.startswith("#Dx:"):
                Recording.append(f.replace(".hea",""))
                line = line.replace('#Dx: ','')
                line = line.replace('\n','')
                labels = line.split(',')
                l = st_df.loc[int(labels[0])]['Abbreviation']
                First_label.append(l)
                if len(labels) > 1:
                    l = st_df.loc[int(labels[1])]['Abbreviation']
                    Second_label.append(l)
                else:
                    Second_label.append('')

                if len(labels) > 2:
                    l = st_df.loc[int(labels[2])]['Abbreviation']
                    Third_label.append(l)
                else:
                    Third_label.append('')

                if len(labels) > 3:
                    l = st_df.loc[int(labels[3])]['Abbreviation']
                    Fourth_label.append(l)
                else:
                    Fourth_label.append('')

                if len(labels) > 4:
                    l = st_df.loc[int(labels[4])]['Abbreviation']
                    Fifth_label.append(l)
                else:
                    Fifth_label.append('')

                if len(labels) > 5:
                    l = st_df.loc[int(labels[5])]['Abbreviation']
                    Fifth_label.append(l)
                else:
                    Fifth_label.append('')

                if len(labels) > 6:
                    l = st_df.loc[int(labels[6])]['Abbreviation']
                    Sixth_label.append(l)
                else:
                    Sixth_label.append('')

                if len(labels) > 7:
                    l = st_df.loc[int(labels[7])]['Abbreviation']
                    Seventh_label.append(l)
                else:
                    Seventh_label.append('')

                if len(labels) > 8:
                    l = st_df.loc[int(labels[8])]['Abbreviation']
                    Eighth_label.append(l)
                else:
                    Eighth_label.append('')

                if len(labels) > 9:
                    l = st_df.loc[int(labels[9])]['Abbreviation']
                    Ninth_label.append(l)
                else:
                    Ninth_label.append('')

                if len(labels) > 10:
                    l = st_df.loc[int(labels[10])]['Abbreviation']
                    Tenth_label.append(l)
                else:
                    Tenth_label.append('')

                if len(labels) > 11:
                    l = st_df.loc[int(labels[11])]['Abbreviation']
                    Eleventh_label.append(l)
                else:
                    Eleventh_label.append('')

                if len(labels) > len_l :
                    len_l = len(labels)

df = pd.DataFrame(zip(Recording, First_label, Second_label, Third_label, Fourth_label, Fifth_label, Sixth_label,
                      Seventh_label, Eighth_label, Ninth_label, Tenth_label, Eleventh_label),
                  columns=['Recording', 'First_label', 'Second_label', 'Third_label', 'Fourth_label',
                           'Fifth_label', 'Sixth_label', 'Seventh_label', 'Eighth_label', 'Ninth_label', 'Tenth_label', 'Eleventh_label'])
df.to_csv(root_path + '/' + target + '_label.csv', index=False)

print(len_l)