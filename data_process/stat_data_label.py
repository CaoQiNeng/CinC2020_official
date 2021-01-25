from include import *

label_path = DATA_ROOT_PATH + '/CinC2020_V1/PhysioNetChallenge2020_PTB-XL_label_SNOMED.csv'
nine_labels = [164889003, 270492004, 164909002, 426783006, 59118001, 284470004,  164884008,  429622005, 164931005]

df = pd.read_csv(label_path)

cc = 0
for i in range(len(df)):
    one_data_labels = df.iloc[i]
    c = 0
    for i in range(len(one_data_labels)):
        if one_data_labels[i] in nine_labels:
            c += 1

    if c == 0:
        cc += 1
        print(one_data_labels[0])

print(cc)
