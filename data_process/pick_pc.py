from include import *

DATA_DIR = DATA_ROOT_PATH + '/CinC2020'
df = pd.read_csv(DATA_DIR + '/cinc2020_label.csv')
plot_path = DATA_DIR + '/PC_plot/Training_WFDB'
for i in range(1, 8):
    os.makedirs(plot_path + '_PAC_plot_' + str(i), exist_ok=True)
    os.makedirs(plot_path + '_PVC_plot_' + str(i), exist_ok=True)

for i, row in enumerate(df.values):
    print(i)
    # index= df.index[i]
    if 'PAC' in row:
        for i in range(1, 8):
            shutil.copy(DATA_DIR + '/Training_WFDB_plot_' + str(i) + '/' + row[0] + '_0.png',
                        plot_path + '_PAC_plot_' + str(i) + '/' + row[0] + '_0.png')

    if 'PVC' in row:
        for i in range(1, 8):
            shutil.copy(DATA_DIR + '/Training_WFDB_plot_' + str(i) + '/' + row[0] + '_0.png',
                        plot_path + '_PVC_plot_' + str(i) + '/' + row[0] + '_0.png')