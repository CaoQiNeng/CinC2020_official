from dataset_32cls_60s import *

def out_result(valid_loader, valid_predict, out_dir):
    probability = valid_predict
    predict = np.array(valid_predict > 0.5, dtype=np.int)
    tabel = []

    for t, (input, truth, infor) in enumerate(valid_loader.dataset):
        temp = []
        temp.append(infor.ecg_id)

        for i in range(len(truth)):
            temp.append(truth[i])

        for i in range(len(truth)):
            temp.append(predict[t][i])

        for i in range(len(truth)):
            temp.append(probability[t][i])

        tabel.append(temp)

    columns = ['ids']
    columns_truth = ['truth_' + class_map_a[i] for i in range(len(class_map))]
    columns_predict = ['predict_' + class_map_a[i] for i in range(len(class_map))]
    columns_probability = ['probability_' + class_map_a[i] for i in range(len(class_map))]

    columns.extend(columns_truth)
    columns.extend(columns_predict)
    columns.extend(columns_probability)

    df = pd.DataFrame(tabel, columns=columns)

    df.to_csv(out_dir + '/result.csv', index=False)

