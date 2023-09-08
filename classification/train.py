import random
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import *
from sklearn import metrics
import pylab as plt

import warnings
from sklearn.utils import shuffle
warnings.filterwarnings("ignore")

def compute_accuracy(predict, label):
    correct_num = 0
    for i in range(len(predict)):
        if predict[i] == label[i]:
            correct_num += 1
    return correct_num / len(predict)

def compute_precision(predict, label):
    top = 0
    bottom = 0
    for i in range(len(label)):
        if predict[i] > 0:
            bottom += 1
            if label[i] > 0:
                top += 1
    if bottom != 0:
        return top / bottom
    else:
        return 0.

def compute_specificity(predict, label):
    top = 0
    bottom = 0
    for i in range(len(label)):
        if label[i] == 0:
            bottom += 1
            if predict[i] == 0:
                top += 1
    if bottom != 0:
        return top / bottom
    else:
        return 0.

def compute_recall(predict, label):
    top = 0
    bottom = 0
    for i in range(len(label)):
        if label[i] > 0:
            bottom += 1
            if predict[i] > 0:
                top += 1
    if bottom != 0:
        return top / bottom
    else:
        return 0.

def compute_Fbeta_score(precision, recall, beta=1):
    top = (1 + beta**2) * precision * recall
    bottom = beta**2 * precision + recall
    if bottom != 0:
        return top / bottom
    else:
        return 0.

def Threshold(y_test, pred, pred_pro):
    pre_true=[]
    pre_false = []
    for i in range(len(y_test)):
        if pred[i] == 1:
            pre_true.append(pred_pro[i])
        if pred[i] == 0:
            pre_false.append(pred_pro[i])
    return (max(pre_false)+min(pre_true))/2

def onehot_enc(y):
    y = np.array(y)
    y = y.reshape(len(y), 1)
    onehot_data = OneHotEncoder(sparse=False)
    onehot_data = onehot_data.fit_transform(y)
    # onehot_data = np.flip(onehot_data, 1)
    return onehot_data

def ROC_AUC(y_test, pred, name='Validation',Fontsize=25):
    plt.figure(figsize=(12, 10))
    plt.title(name+' ROC',fontsize=Fontsize)
    color = ['r', 'g', 'b', 'y', 'c']
    for lists in range(len(pred)):
        fpr, tpr, threshold = metrics.roc_curve(y_test[lists], pred[lists],drop_intermediate=True)
        roc_auc = metrics.auc(fpr, tpr)
        plt.plot(fpr, tpr, color[lists], label='%s%s AUC = %0.3f' % (name, lists+1, roc_auc))

    plt.legend(loc='lower right', fontsize=20)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.tick_params(labelsize=13)
    plt.ylabel('True Positive Rate',fontsize=Fontsize)
    plt.xlabel('False Positive Rate',fontsize=Fontsize)
    plt.show()

def predict_evaluate(lr,lr2,X_test,X_test2,y_test,fold_num):
    # T1
    val_output = lr.predict(X_test)
    val_score = compute_accuracy(val_output, y_test)
    val_p = compute_precision(val_output, y_test)
    val_r = compute_recall(val_output, y_test)
    val_f1 = compute_Fbeta_score(val_p, val_r, 1)
    val_sp = compute_specificity(val_output, y_test)
    val_auc = roc_auc_score(onehot_enc(y_test), lr.predict_proba(X_test))
    print('T1_fold_%s:' % fold_num)
    print('VAL accuracy: {:.3f}'.format(val_score))
    print('VAL precision: {:.3f}'.format(val_p))
    print('VAL recall: {:.3f}'.format(val_r))
    print('VAL f1: {:.3f}'.format(val_f1))
    print('VAL specificity: {:.3f}'.format(val_sp))
    print('VAL AUC: {:.3f}'.format(val_auc))

    lr_pred = lr.predict_proba(X_test)[:, 1]
    T1_threshold_value = Threshold(y_test, val_output, lr_pred)

    # T1C
    val_output2 = lr2.predict(X_test2)
    val_score2 = compute_accuracy(val_output2, y_test)
    val_p2 = compute_precision(val_output2, y_test)
    val_r2 = compute_recall(val_output2, y_test)
    val_f12 = compute_Fbeta_score(val_p2, val_r2, 1)
    val_sp2 = compute_specificity(val_output2, y_test)
    val_auc2 = roc_auc_score(onehot_enc(y_test), lr2.predict_proba(X_test2))
    print('T1C_fold_%s:' % fold_num)
    print('VAL accuracy: {:.3f}'.format(val_score2))
    print('VAL precision: {:.3f}'.format(val_p2))
    print('VAL recall: {:.3f}'.format(val_r2))
    print('VAL f1: {:.3f}'.format(val_f12))
    print('VAL specificity: {:.3f}'.format(val_sp2))
    print('VAL AUC: {:.3f}'.format(val_auc2))

    lr_pred2 = lr2.predict_proba(X_test2)[:, 1]
    T1C_threshold_value = Threshold(y_test, val_output2, lr_pred2)

    # T1+T1C
    val_output_3_pro = (lr.predict_proba(X_test) + lr2.predict_proba(X_test2)) / 2
    lr_pred3 = val_output_3_pro[:, 1]

    _threshold_value = (T1_threshold_value + T1C_threshold_value) / 2
    val_output3 = []
    for pro in val_output_3_pro:
        if pro[1] >= _threshold_value + 0.000:
            val_output3.append(1)
        else:
            val_output3.append(0)
    val_score3 = compute_accuracy(val_output3, y_test)
    val_p3 = compute_precision(val_output3, y_test)
    val_r3 = compute_recall(val_output3, y_test)
    val_f13 = compute_Fbeta_score(val_p3, val_r3, 1)
    val_sp3 = compute_specificity(val_output3, y_test)
    val_auc3 = roc_auc_score(onehot_enc(y_test), val_output_3_pro)
    print('T1+T1C_fold_%s:' % fold_num)
    print('VAL accuracy: {:.3f}'.format(val_score3))
    print('VAL precision: {:.3f}'.format(val_p3))
    print('VAL recall: {:.3f}'.format(val_r3))
    print('VAL f1: {:.3f}'.format(val_f13))
    print('VAL specificity: {:.3f}'.format(val_sp3))
    print('VAL AUC: {:.3f}'.format(val_auc3))

    return lr_pred,lr_pred2,lr_pred3,val_auc,val_score,val_sp,val_r,val_auc2,val_score2,val_sp2,val_r2,val_auc3,val_score3,val_sp3,val_r3


def five_fold_validation(excel_file=r'D:\Radiomics and Deep Learning for Identification and Automated Delineation\train_T1.xlsx',
                         excel_file2=r'D:\Radiomics and Deep Learning for Identification and Automated Delineation\train_T1C.xlsx',
                         feature_T1=[105, 6, 207, 9, 841, 773, 12, 436, 479, 46],
                         feature_T1C = [105, 68, 760, 755, 390, 773, 9, 206, 44, 828],
                         partition_min=12,
                         partition_max=18,
                         plot_ROC=True
                         ):
    df = pd.read_excel(excel_file)
    df2 = pd.read_excel(excel_file2)

    features_origin = pd.DataFrame(df.values)
    features_origin2 = pd.DataFrame(df2.values)

    #Feature columns in the excel files
    feature_ID = np.array(feature_T1)+2
    feature_ID2 = np.array(feature_T1C)+2

    time = 100
    p = 5
    bad_fold=0
    true_fold=0
    KF = KFold(n_splits=p)

    for times in range(time):
        Random_seed = random.randint(0, time)
        features_origin = shuffle(features_origin, random_state=Random_seed)
        features_origin2 = shuffle(features_origin2, random_state=Random_seed)

        fea_number = len(feature_ID)
        features = np.zeros(shape=(features_origin.shape[0], fea_number))
        for i in range(fea_number):
            features[:, i] = features_origin[features_origin.columns[feature_ID[i]]]

        fea_number2 = len(feature_ID2)
        features2 = np.zeros(shape=(features_origin2.shape[0], fea_number2))
        for i in range(fea_number2):
            features2[:, i] = features_origin2[features_origin2.columns[feature_ID2[i]]]

        labels = features_origin[features_origin.columns[1]]

        is_bad_fold = False
        AUC_results_svm = []
        AUC_results_rf = []
        AUC_results_xgb = []
        ACC_results_svm = []
        ACC_results_rf = []
        ACC_results_xgb = []
        SP_results_svm = []
        SP_results_rf = []
        SP_results_xgb = []
        RC_results_svm = []
        RC_results_rf = []
        RC_results_xgb = []

        AUC_results_svm2 = []
        AUC_results_rf2 = []
        AUC_results_xgb2 = []
        ACC_results_svm2 = []
        ACC_results_rf2 = []
        ACC_results_xgb2 = []
        SP_results_svm2 = []
        SP_results_rf2 = []
        SP_results_xgb2 = []
        RC_results_svm2 = []
        RC_results_rf2 = []
        RC_results_xgb2 = []

        AUC_results_svm3 = []
        AUC_results_rf3 = []
        AUC_results_xgb3 = []
        ACC_results_svm3 = []
        ACC_results_rf3 = []
        ACC_results_xgb3 = []
        SP_results_svm3 = []
        SP_results_rf3 = []
        SP_results_xgb3 = []
        RC_results_svm3 = []
        RC_results_rf3 = []
        RC_results_xgb3 = []

        for train_index, test_index in KF.split(features):
            y_train, y_test = labels.iloc[train_index], labels.iloc[test_index]
            # We want a small fluctuation in the ratio of positive and negative samples for each fold
            if np.sum(y_test) <= partition_min or np.sum(y_test) >= partition_max:
                bad_fold += 1
                is_bad_fold = True
                break
        pred_all = []
        label_all = []
        pred_all2 = []
        pred_all3 = []
        fold_num=0
        if not is_bad_fold:
            true_fold+=1
            for train_index, test_index in KF.split(features):
                fold_num+=1
                X_train, X_test = features[train_index], features[test_index]
                X_train2, X_test2 = features2[train_index], features2[test_index]
                y_train, y_test = labels.iloc[train_index], labels.iloc[test_index]
                # print('start sum of (y_test)', np.sum(y_test))
                stand = StandardScaler()

                X_train = stand.fit_transform(X_train)
                X_test = stand.transform(X_test)
                X_train2 = stand.fit_transform(X_train2)
                X_test2 = stand.transform(X_test2)

                X_train = np.array(X_train)
                X_test = np.array(X_test)
                X_train2 = np.array(X_train2)
                X_test2 = np.array(X_test2)
                y_train = np.array(y_train)
                y_test = np.array(y_test)

                for model_num in range(3):
                    if model_num == 0:
                        print('SVM：')
                        lr = SVC(C=1, cache_size=2000, class_weight='balanced', coef0=0.0,
                                 decision_function_shape='ovr', degree=3, gamma='auto', kernel='linear',
                                 max_iter=30000, probability=True, random_state=None, shrinking=True,
                                 tol=0.001, verbose=False).fit(X_train, y_train)#T1
                        lr2 = SVC(C=1, cache_size=2000, class_weight='balanced', coef0=0.0,
                                 decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
                                 max_iter=30000, probability=True, random_state=None, shrinking=True,
                                 tol=0.001, verbose=False).fit(X_train2, y_train)#T1C
                        lr_pred_T1, lr_pred_T1C, lr_pred_ensemble, val_auc_T1, val_score_T1, val_sp_T1, val_r_T1, \
                        val_auc_T1C, val_score_T1C, val_sp_T1C, val_r_T1C, val_auc_ensemble, val_score_ensemble, \
                        val_sp_ensemble, val_r_ensemble=predict_evaluate(lr,lr2,X_test,X_test2,y_test,fold_num)
                        #for plot
                        pred_all.append(lr_pred_T1)
                        pred_all2.append(lr_pred_T1C)
                        pred_all3.append(lr_pred_ensemble)
                        label_all.append(y_test)
                        #
                        AUC_results_svm.append(val_auc_T1)
                        ACC_results_svm.append(val_score_T1)
                        SP_results_svm.append(val_sp_T1)
                        RC_results_svm.append(val_r_T1)
                        AUC_results_svm2.append(val_auc_T1C)
                        ACC_results_svm2.append(val_score_T1C)
                        SP_results_svm2.append(val_sp_T1C)
                        RC_results_svm2.append(val_r_T1C)
                        AUC_results_svm3.append(val_auc_ensemble)
                        ACC_results_svm3.append(val_score_ensemble)
                        SP_results_svm3.append(val_sp_ensemble)
                        RC_results_svm3.append(val_r_ensemble)

                    if model_num == 1:
                        print('RF：')
                        lr = RandomForestClassifier(max_depth=10, n_estimators=100, max_features='auto', criterion='gini',
                                                    min_samples_leaf=2, min_samples_split=3, bootstrap=True, class_weight='balanced').fit(X_train,
                                                                                                                 y_train)  # T1
                        lr2 = RandomForestClassifier(max_depth=10, n_estimators=100, max_features='auto', criterion='gini',
                                                    min_samples_leaf=1, min_samples_split=2, bootstrap=True, class_weight='balanced').fit(X_train2,
                                                                                                                 y_train)  # T1C
                        lr_pred_T1, lr_pred_T1C, lr_pred_ensemble, val_auc_T1, val_score_T1, val_sp_T1, val_r_T1, \
                        val_auc_T1C, val_score_T1C, val_sp_T1C, val_r_T1C, val_auc_ensemble, val_score_ensemble, \
                        val_sp_ensemble, val_r_ensemble = predict_evaluate(lr, lr2, X_test, X_test2, y_test, fold_num)
                        AUC_results_rf.append(val_auc_T1)
                        ACC_results_rf.append(val_score_T1)
                        SP_results_rf.append(val_sp_T1)
                        RC_results_rf.append(val_r_T1)
                        AUC_results_rf2.append(val_auc_T1C)
                        ACC_results_rf2.append(val_score_T1C)
                        SP_results_rf2.append(val_sp_T1C)
                        RC_results_rf2.append(val_r_T1C)
                        AUC_results_rf3.append(val_auc_ensemble)
                        ACC_results_rf3.append(val_score_ensemble)
                        SP_results_rf3.append(val_sp_ensemble)
                        RC_results_rf3.append(val_r_ensemble)
                    if model_num == 2:
                        print('XGB：')
                        lr = XGBClassifier(eta=0.05,
                                           gamma=0.2,
                                           max_depth=9,
                                           min_child_weight=0.0,
                                           subsample=0.7,
                                           colsample_bytree=0.8,
                                           alpha=0.0,
                                           scale_pos_weight=7).fit(X_train, y_train)  # T1
                        lr2 = XGBClassifier(eta=0.05,
                                            gamma=0.2,
                                            max_depth=9,
                                            min_child_weight=0.0,
                                            subsample=0.7,
                                            colsample_bytree=0.8,
                                            alpha=0.0,
                                            scale_pos_weight=7).fit(X_train2, y_train)  # T1C
                        lr_pred_T1, lr_pred_T1C, lr_pred_ensemble, val_auc_T1, val_score_T1, val_sp_T1, val_r_T1, \
                        val_auc_T1C, val_score_T1C, val_sp_T1C, val_r_T1C, val_auc_ensemble, val_score_ensemble, \
                        val_sp_ensemble, val_r_ensemble = predict_evaluate(lr, lr2, X_test, X_test2, y_test, fold_num)
                        AUC_results_xgb.append(val_auc_T1)
                        ACC_results_xgb.append(val_score_T1)
                        SP_results_xgb.append(val_sp_T1)
                        RC_results_xgb.append(val_r_T1)
                        AUC_results_xgb2.append(val_auc_T1C)
                        ACC_results_xgb2.append(val_score_T1C)
                        SP_results_xgb2.append(val_sp_T1C)
                        RC_results_xgb2.append(val_r_T1C)
                        AUC_results_xgb3.append(val_auc_ensemble)
                        ACC_results_xgb3.append(val_score_ensemble)
                        SP_results_xgb3.append(val_sp_ensemble)
                        RC_results_xgb3.append(val_r_ensemble)

            print('T1')
            print('SVM', AUC_results_svm)
            print(sum(ACC_results_svm) / len(ACC_results_svm))
            print(sum(SP_results_svm) / len(SP_results_svm))
            print(sum(RC_results_svm) / len(RC_results_svm))
            print(sum(AUC_results_svm)/len(AUC_results_svm))
            print('RF', AUC_results_rf)
            print(sum(ACC_results_rf) / len(ACC_results_rf))
            print(sum(SP_results_rf) / len(SP_results_rf))
            print(sum(RC_results_rf) / len(RC_results_rf))
            print(sum(AUC_results_rf)/len(AUC_results_rf))
            print('XGB', AUC_results_xgb)
            print(sum(ACC_results_xgb) / len(ACC_results_xgb))
            print(sum(SP_results_xgb) / len(SP_results_xgb))
            print(sum(RC_results_xgb) / len(RC_results_xgb))
            print(sum(AUC_results_xgb)/len(AUC_results_xgb))
            print('-'*100)
            print('T1C')
            print('SVM', AUC_results_svm2)
            print(sum(ACC_results_svm2) / len(ACC_results_svm2))
            print(sum(SP_results_svm2) / len(SP_results_svm2))
            print(sum(RC_results_svm2) / len(RC_results_svm2))
            print(sum(AUC_results_svm2)/len(AUC_results_svm2))
            print('RF', AUC_results_rf2)
            print(sum(ACC_results_rf2) / len(ACC_results_rf2))
            print(sum(SP_results_rf2) / len(SP_results_rf2))
            print(sum(RC_results_rf2) / len(RC_results_rf2))
            print(sum(AUC_results_rf2)/len(AUC_results_rf2))
            print('XGB', AUC_results_xgb2)
            print(sum(ACC_results_xgb2) / len(ACC_results_xgb2))
            print(sum(SP_results_xgb2) / len(SP_results_xgb2))
            print(sum(RC_results_xgb2) / len(RC_results_xgb2))
            print(sum(AUC_results_xgb2)/len(AUC_results_xgb2))
            print('T1+T1C')
            print('SVM', AUC_results_svm3)
            print(sum(ACC_results_svm3) / len(ACC_results_svm3))
            print(sum(SP_results_svm3) / len(SP_results_svm3))
            print(sum(RC_results_svm3) / len(RC_results_svm3))
            print(sum(AUC_results_svm3) / len(AUC_results_svm3))
            print('RF', AUC_results_rf3)
            print(sum(ACC_results_rf3) / len(ACC_results_rf3))
            print(sum(SP_results_rf3) / len(SP_results_rf3))
            print(sum(RC_results_rf3) / len(RC_results_rf3))
            print(sum(AUC_results_rf3)/len(AUC_results_rf3))
            print('XGB', AUC_results_xgb3)
            print(sum(ACC_results_xgb3) / len(ACC_results_xgb3))
            print(sum(SP_results_xgb3) / len(SP_results_xgb3))
            print(sum(RC_results_xgb3) / len(RC_results_xgb3))
            print(sum(AUC_results_xgb3)/len(AUC_results_xgb3))
            print('-'*100)
            # plot ROC
            if plot_ROC==True:
                ROC_AUC(label_all, pred_all, name='T1 five-fold Validation',Fontsize=25)
                ROC_AUC(label_all, pred_all2, name='T1C five-fold Validation',Fontsize=25)
                ROC_AUC(label_all, pred_all3, name='T1+T1C five-fold Validation',Fontsize=25)
            break

if __name__ == '__main__':
    five_fold_validation(excel_file=r'D:\Radiomics and Deep Learning for Identification and Automated Delineation\train_T1.xlsx',
                         excel_file2=r'D:\Radiomics and Deep Learning for Identification and Automated Delineation\train_T1C.xlsx',
                         feature_T1=[105, 6, 207, 9, 841, 773, 12, 436, 479, 46],
                         feature_T1C=[105, 68, 760, 755, 390, 773, 9, 206, 44, 828]
                         )