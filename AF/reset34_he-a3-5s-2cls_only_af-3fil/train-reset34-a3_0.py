import os
os.environ['CUDA_VISIBLE_DEVICES']='0'

from common  import *
from model_resnet34_he import *
from dataset_only_af_5s_no_merge_fil_proton import *
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score

def compute_beta_score(labels, output, beta, num_classes, check_errors=True):
    # Check inputs for errors.
    if check_errors:
        if len(output) != len(labels):
            raise Exception('Numbers of outputs and labels must be the same.')

    # Populate contingency table.
    num_recordings = len(labels)

    fbeta_l = np.zeros(num_classes)
    gbeta_l = np.zeros(num_classes)
    fmeasure_l = np.zeros(num_classes)
    accuracy_l = np.zeros(num_classes)

    f_beta = 0
    g_beta = 0
    f_measure = 0
    accuracy = 0

    # Weight function
    C_l = np.ones(num_classes);

    for j in range(num_classes):
        tp = 0
        fp = 0
        fn = 0
        tn = 0

        for i in range(num_recordings):

            num_labels = np.sum(labels[i])

            if labels[i][j] and output[i][j]:
                tp += 1 / num_labels
            elif not labels[i][j] and output[i][j]:
                fp += 1 / num_labels
            elif labels[i][j] and not output[i][j]:
                fn += 1 / num_labels
            elif not labels[i][j] and not output[i][j]:
                tn += 1 / num_labels

        # Summarize contingency table.
        if ((1 + beta ** 2) * tp + (fn * beta ** 2) + fp):
            fbeta_l[j] = float((1 + beta ** 2) * tp) / float(((1 + beta ** 2) * tp) + (fn * beta ** 2) + fp)
        else:
            fbeta_l[j] = 1.0

        if (tp + fp + beta * fn):
            gbeta_l[j] = float(tp) / float(tp + fp + beta * fn)
        else:
            gbeta_l[j] = 1.0

        if tp + fp + fn + tn:
            accuracy_l[j] = float(tp + tn) / float(tp + fp + fn + tn)
        else:
            accuracy_l[j] = 1.0

        if 2 * tp + fp + fn:
            fmeasure_l[j] = float(2 * tp) / float(2 * tp + fp + fn)
        else:
            fmeasure_l[j] = 1.0

    for i in range(num_classes):
        f_beta += fbeta_l[i] * C_l[i]
        g_beta += gbeta_l[i] * C_l[i]
        f_measure += fmeasure_l[i] * C_l[i]
        accuracy += accuracy_l[i] * C_l[i]

    f_beta = float(f_beta) / float(num_classes)
    g_beta = float(g_beta) / float(num_classes)
    f_measure = float(f_measure) / float(num_classes)
    accuracy = float(accuracy) / float(num_classes)

    return accuracy, f_measure, f_beta, g_beta

#https://stackoverflow.com/questions/43162506/undefinedmetricwarning-f-score-is-ill-defined-and-being-set-to-0-0-in-labels-wi
def metric(truth, predict):
    truth_for_cls = np.sum(truth, axis=0) + 1e-11
    predict_for_cls = np.sum(predict, axis=0) + 1e-11

    # TP
    count = truth + predict
    count[count != 2] = 0
    TP = np.sum(count, axis=0) / 2

    precision = TP / predict_for_cls
    recall = TP / truth_for_cls

    return precision, recall

################################################################################################
#------------------------------------
def do_valid(net, valid_loader, out_dir=None):
    def out_result(valid_loader, valid_predict):
        probability = valid_predict
        predict = np.array(valid_predict > 0.5, dtype=np.int)
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
                                   truth_PAC,truth_PVC,truth_STD,truth_STE,
                                   predict_AF, predict_I_AVB, predict_LBBB, predict_Normal, predict_RBBB,
                                   predict_PAC,predict_PVC,predict_STD,predict_STE,
                                   probability_AF, probability_I_AVB, probability_LBBB, probability_Normal,
                                   probability_RBBB, probability_PAC,probability_PVC,probability_STD,
                                   probability_STE),
                          columns=['ids',
                                   'truth_AF', 'truth_I_AVB', 'truth_LBBB', 'truth_Normal',
                                   'truth_PAC','truth_PVC', 'truth_RBBB', 'truth_STD','truth_STE',
                                   'predict_AF', 'predict_I_AVB', 'predict_LBBB', 'predict_Normal',
                                   'predict_PAC','predict_PVC', 'predict_RBBB','predict_STD','predict_STE',
                                   'probability_AF', 'probability_I_AVB', 'probability_LBBB', 'probability_Normal',
                                   'probability_PAC','probability_PVC', 'probability_RBBB', 'probability_STD',
                                   'probability_STE'])

        df.to_csv(out_dir + '/result.csv', index=False)

    valid_loss = 0
    valid_predict = []
    valid_truth = []
    valid_num = 0

    for t, (input, truth, infor) in enumerate(valid_loader):
        batch_size = len(infor)

        net.eval()
        input = input.unsqueeze(3)
        input  = input.cuda()
        truth  = truth.cuda()

        with torch.no_grad():
            logit = data_parallel(net, input) #net(input)
            probability = torch.sigmoid(logit)

            probability[:, 1:] = 0
            truth[:, 1:] = 0

            loss = F.binary_cross_entropy(probability, truth)

        valid_predict.append(probability.cpu().numpy())
        valid_truth.append(truth.cpu().numpy().astype(int))

        #---
        valid_loss += loss.cpu().numpy() * batch_size

        valid_num  += batch_size

        print('\r %8d / %d'%(valid_num, len(valid_loader.dataset)),end='',flush=True)

        pass  #-- end of one data loader --

    assert(valid_num == len(valid_loader.dataset))
    valid_loss = valid_loss / (valid_num+1e-8)

    valid_truth = np.vstack(valid_truth)
    valid_predict = np.vstack(valid_predict)

    if 0:
        out_result(valid_loader, valid_predict)

    accuracy, f_measure, f_beta, g_beta = compute_beta_score(valid_truth, valid_predict>0.5, 2, valid_truth.shape[1], check_errors=True)
    valid_precision, valid_recall = metric(valid_truth, (valid_predict>0.5).astype(int))

    return [accuracy, f_beta,g_beta,f_measure], valid_loss, valid_precision, valid_recall

def run_train():
    train_fold = 3
    valid_fold = 0
    out_dir = ROOT_PATH + '/CinC2020_official_logs/result-reset34_he-a%d_%d-5s-2cls_only_af-3fil'%(train_fold, valid_fold)
    initial_checkpoint = None
    # initial_checkpoint = ROOT_PATH + '/CinC2020_official_logs/result-reset34-a%d_%d-5s-2cls_only_af-fil_proton/checkpoint/00013200_model.pth'%(train_fold, valid_fold)

    schduler = NullScheduler(lr=0.1)
    iter_accum = 1
    batch_size = 16 #8

    ## setup  -----------------------------------------------------------------------------
    for f in ['checkpoint','train','valid','backup'] : os.makedirs(out_dir +'/'+f, exist_ok=True)
    backup_project_as_zip(PROJECT_PATH, out_dir +'/backup/code.train.%s.zip'%IDENTIFIER)

    log = Logger()
    log.open(out_dir+'/log.train.txt',mode='a')
    log.write('\n--- [START %s] %s\n\n' % (IDENTIFIER, '-' * 64))
    log.write('\t%s\n' % COMMON_STRING)
    log.write('\n')

    log.write('\tSEED         = %u\n' % SEED)
    log.write('\tPROJECT_PATH = %s\n' % PROJECT_PATH)
    log.write('\t__file__     = %s\n' % __file__)
    log.write('\tout_dir      = %s\n' % out_dir)
    log.write('\n')

    ## dataset ----------------------------------------
    log.write('** dataset setting **\n')

    train_dataset = CinCDataset(
        mode='train',
        csv='train.csv',
        split='valid_a%d_687.npy' % 0,
        data_path=DATA_DIR + '/data_argument/5s_3fil_a3_0/train_data'
    )
    train_loader = DataLoader(
        train_dataset,
        # sampler     = RandomSampler(train_dataset),
        shuffle=True,
        batch_size=batch_size,
        drop_last=False,
        num_workers=20,
        pin_memory=True,
        collate_fn=null_collate
    )

    val_dataset = CinCDataset(
        mode='train',
        csv='train.csv',
        split='valid_a%d_687.npy' % 0,
        data_path=DATA_DIR + '/data_argument/5s_3fil_a3_0/valid_data'
    )
    valid_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        drop_last=False,
        num_workers=10,
        pin_memory=True,
        collate_fn=null_collate
    )

    assert(len(train_dataset)>=batch_size)
    log.write('batch_size = %d\n'%(batch_size))
    log.write('train_dataset : \n%s\n'%(train_dataset))
    log.write('valid_dataset : \n%s\n'%(val_dataset))
    log.write('\n')

    ## net ----------------------------------------
    log.write('** net setting **\n')
    net = resnet34().cuda()
    log.write('\tinitial_checkpoint = %s\n' % initial_checkpoint)

    if initial_checkpoint is not None:
        state_dict = torch.load(initial_checkpoint, map_location=lambda storage, loc: storage)

        # net.load_state_dict(state_dict,strict=False)
        net.load_state_dict(state_dict,strict=True)  #True


    log.write('net=%s\n'%(type(net)))
    log.write('\n')

    #optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()),lr=schduler(0))
    #optimizer = torch.optim.RMSprop(net.parameters(), lr =0.0005, alpha = 0.95)
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=schduler(0), momentum=0.0, weight_decay=0.0)

    num_iters   = 3000*1000
    iter_smooth = 200
    iter_log    = 200
    iter_valid  = 200
    iter_save   = [0, num_iters-1]\
                   + list(range(0, num_iters, 200))#1*1000

    start_iter = 0
    start_epoch= 0
    rate       = 0
    if initial_checkpoint is not None:
        initial_optimizer = initial_checkpoint.replace('_model.pth','_optimizer.pth')
        if os.path.exists(initial_optimizer):
            checkpoint  = torch.load(initial_optimizer)
            start_iter  = checkpoint['iter' ]
            start_epoch = checkpoint['epoch']
            #optimizer.load_state_dict(checkpoint['optimizer'])
        pass

    log.write('optimizer\n  %s\n'%(optimizer))
    log.write('schduler\n  %s\n'%(schduler))
    log.write('\n')

    ## start training here! ##############################################
    log.write('** start training here! **\n')
    log.write('   batch_size=%d,  iter_accum=%d\n'%(batch_size,iter_accum))
    log.write('   experiment  = %s\n' % str(__file__.split('/')[-2:]))
    log.write('----------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------\n')
    log.write('mode    rate    iter  epoch | CinC                    | loss  | AF        | I-AVB     | LBBB      | Normal    | PAC       | PVC       | RBBB      | STD       | STE       | time        \n')
    log.write('----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------\n')
              # train  0.01000   0.5   0.2 | 0.648 0.508 0.830 0.830 | 1.11  | 1.11 0.29 | 1.11 0.29 | 1.11 0.29 | 1.11 0.29 | 1.11 0.29 | 1.11 0.29 | 1.11 0.29 | 1.11 0.29 | 1.11 0.29 | 0 hr 05 min
    def message(rate, iter, epoch, CinC, loss, precision, recall, mode='print', train_mode = 'train'):
        precision_recall = []
        for p, r in zip(precision, recall) :
            precision_recall.append(p)
            precision_recall.append(r)

        if mode==('print'):
            asterisk = ' '
        if mode==('log'):
            asterisk = '*' if iter in iter_save else ' '

        text = \
            '%s   %0.5f %5.1f%s %4.1f | '%(train_mode, rate, iter/1000, asterisk, epoch,) +\
            '%0.3f %0.3f %0.3f %0.3f | '%(*CinC, ) +\
            '%4.3f | '%loss +\
            '%0.2f %0.2f | %0.2f %0.2f | %0.2f %0.2f | %0.2f %0.2f | %0.2f %0.2f | %0.2f %0.2f | %0.2f %0.2f | %0.2f %0.2f | %0.2f %0.2f | '%(*precision_recall, ) +\
            '%s' % (time_to_str((timer() - start_timer),'min'))

        return text

    #----
    CinC = (0, 0, 0)
    train_loss = 0
    train_precision = [0 for i in range(len(class_map))]
    train_recall = [0 for i in range(len(class_map))]
    train_accuracy = 0
    train_f_beta = 0
    train_g_beta = 0
    train_f_measure = 0
    iter = 0
    i    = 0

    start_timer = timer()
    while  iter<num_iters:
        train_predict_list = []
        train_truth_list = []
        sum_train_loss = 0
        sum_train = 0

        optimizer.zero_grad()
        for t, (input, truth, infor) in enumerate(train_loader):
            batch_size = len(infor)
            iter  = i + start_iter
            epoch = (iter-start_iter)*batch_size/len(train_dataset) + start_epoch

            #if 0:
            if (iter % iter_valid==0):
                CinC, valid_loss, valid_precision, valid_recall = do_valid(net, valid_loader, out_dir) #
                pass

            if (iter % iter_log==0):
                print('\r',end='',flush=True)
                print(message(rate, iter, epoch, [train_accuracy, train_f_beta, train_g_beta, train_f_measure], train_loss, train_precision, train_recall, mode='log', train_mode='train'))
                log.write(message(rate, iter, epoch, CinC, valid_loss, valid_precision, valid_recall,mode='log', train_mode='valid'))
                log.write('\n')

            top_loss = np.array([0.15 for i in range(10)])
            top_F2 = [0.83 for i in range(10)]
            #if 0:
            if iter in iter_save:
                current_loss = valid_loss
                current_F2 = CinC[1]

                top_loss = np.sort(top_loss)
                top_F2 = np.sort(top_F2)

                if current_loss < top_loss[9]:
                    top_loss[9] = current_loss

                if current_F2 > top_F2[0]:
                    top_F2[0] = current_F2

                torch.save({
                    #'optimizer': optimizer.state_dict(),
                    'iter'     : iter,
                    'epoch'    : epoch,
                }, out_dir +'/checkpoint/%08d_optimizer.pth'%(iter))
                if iter!=start_iter:
                    torch.save(net.state_dict(),out_dir +'/checkpoint/%08d_model.pth'%(iter))
                    pass

            # learning rate schduler -------------
            lr = schduler(iter)
            if lr<0 : break
            adjust_learning_rate(optimizer, lr)
            rate = get_learning_rate(optimizer)

            # one iteration update  -------------
            #net.set_mode('train',is_freeze_bn=True)
            net.train()
            input = input.unsqueeze(3)
            input = input.cuda()
            truth = truth.cuda()

            logit = data_parallel(net, input)
            probability = torch.sigmoid(logit)

            loss = F.binary_cross_entropy(probability, truth)

            loss.backward()
            loss = loss.detach().cpu().numpy()
            if (iter % iter_accum)==0:
                optimizer.step()
                optimizer.zero_grad()

            predict = probability.cpu().detach().numpy()
            truth = truth.cpu().numpy().astype(int)
            batch_precision, batch_recall= metric(truth, (predict>0.5).astype(int))
            batch_accuracy, batch_f_measure, batch_f_beta, batch_g_beta = compute_beta_score(truth, predict>0.5, 2,
                                                                     truth.shape[1], check_errors=True)

            # print statistics  --------
            batch_loss      = loss
            train_predict_list.append(predict)
            train_truth_list.append(truth)
            sum_train_loss += loss * batch_size
            sum_train      += batch_size
            if iter%iter_smooth == 0:
                train_loss = sum_train_loss / (sum_train+1e-12)
                train_predict_list = np.vstack(train_predict_list)
                train_truth_list = np.vstack(train_truth_list)
                train_precision, train_recall = metric(train_truth_list, (train_predict_list>0.5).astype(int))
                train_accuracy, train_f_measure, train_f_beta, train_g_beta = compute_beta_score(train_truth_list, train_predict_list>0.5, 2,
                                                                         train_truth_list.shape[1], check_errors=True)
                train_predict_list = []
                train_truth_list = []
                sum_train_loss = 0
                sum_train      = 0

            # print(batch_loss)
            print('\r',end='',flush=True)
            print(message(rate, iter, epoch, [batch_accuracy, batch_f_beta, batch_g_beta, batch_f_measure], batch_loss, batch_precision, batch_recall, mode='log', train_mode='train'), end='',flush=True)
            i=i+1

        pass  #-- end of one data loader --
    pass #-- end of all iterations --
    log.write('\n')

# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    run_train()
