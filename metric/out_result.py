from dataset_25cls_60s import *

def out_result(valid_loader, valid_predict):
    probability = valid_predict
    predict = np.array(valid_predict > 0.5, dtype=np.int)
    tabel = [[] for i in range(len(class_map) + 1)]

    print(tabel)
    exit()


    ids = []
    truth_AF = []
    truth_I_AVB = []
    truth_LBBB = []
    truth_Normal = []
    truth_RBBB = []
    truth_PAC = []
    truth_PVC = []
    truth_STD = []
    truth_STE = []

    predict_AF = []
    predict_I_AVB = []
    predict_LBBB = []
    predict_Normal = []
    predict_RBBB = []
    predict_PAC = []
    predict_PVC = []
    predict_STD = []
    predict_STE = []

    probability_AF = []
    probability_I_AVB = []
    probability_LBBB = []
    probability_Normal = []
    probability_RBBB = []
    probability_PAC = []
    probability_PVC = []
    probability_STD = []
    probability_STE = []

    for t, (input, truth, infor) in enumerate(valid_loader.dataset):
        ids.append(infor.ecg_id)
        truth_AF.append(truth[0])
        truth_I_AVB.append(truth[1])
        truth_LBBB.append(truth[2])
        truth_Normal.append(truth[3])
        truth_RBBB.append(truth[4])
        truth_PAC.append(truth[5])
        truth_PVC.append(truth[6])
        truth_STD.append(truth[7])
        truth_STE.append(truth[8])

        predict_AF.append(predict[t][0])
        predict_I_AVB.append(predict[t][1])
        predict_LBBB.append(predict[t][2])
        predict_Normal.append(predict[t][3])
        predict_RBBB.append(predict[t][4])
        predict_PAC.append(predict[t][5])
        predict_PVC.append(predict[t][6])
        predict_STD.append(predict[t][7])
        predict_STE.append(predict[t][8])

        probability_AF.append(probability[t][0])
        probability_I_AVB.append(probability[t][1])
        probability_LBBB.append(probability[t][2])
        probability_Normal.append(probability[t][3])
        probability_RBBB.append(probability[t][4])
        probability_PAC.append(probability[t][5])
        probability_PVC.append(probability[t][6])
        probability_STD.append(probability[t][7])
        probability_STE.append(probability[t][8])

    df = pd.DataFrame(zip(ids, truth_AF, truth_I_AVB, truth_LBBB, truth_Normal, truth_RBBB,
                          truth_PAC, truth_PVC, truth_STD, truth_STE,
                          predict_AF, predict_I_AVB, predict_LBBB, predict_Normal, predict_RBBB,
                          predict_PAC, predict_PVC, predict_STD, predict_STE,
                          probability_AF, probability_I_AVB, probability_LBBB, probability_Normal,
                          probability_RBBB, probability_PAC, probability_PVC, probability_STD,
                          probability_STE),
                      columns=['ids',
                               'truth_AF', 'truth_I_AVB', 'truth_LBBB', 'truth_Normal',
                               'truth_PAC', 'truth_PVC', 'truth_RBBB', 'truth_STD', 'truth_STE',
                               'predict_AF', 'predict_I_AVB', 'predict_LBBB', 'predict_Normal',
                               'predict_PAC', 'predict_PVC', 'predict_RBBB', 'predict_STD', 'predict_STE',
                               'probability_AF', 'probability_I_AVB', 'probability_LBBB', 'probability_Normal',
                               'probability_PAC', 'probability_PVC', 'probability_RBBB', 'probability_STD',
                               'probability_STE'])

    # df.to_csv(out_dir + '/result.csv', index=False)

out_result(1, 1)