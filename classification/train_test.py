import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.metrics import roc_auc_score
from sklearn import metrics
import pylab as plt
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost.sklearn import XGBClassifier
import warnings

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
def onehot_enc(y):
    y = np.array(y)
    y = y.reshape(len(y), 1)
    onehot_data = OneHotEncoder(sparse=False)
    onehot_data = onehot_data.fit_transform(y)
    # onehot_data = np.flip(onehot_data, 1)
    return onehot_data

def ROC_AUC_3(y_test, pred, pred2, pred3, name='Test',Fontsize=25):
    plt.figure(figsize=(12, 10))
    plt.title(name+' ROC',fontsize=Fontsize)
    color = ['r', 'g', 'b', 'y', 'c']
    fpr, tpr, threshold = metrics.roc_curve(y_test, pred,drop_intermediate=True)
    roc_auc = metrics.auc(fpr, tpr)
    plt.plot(fpr, tpr, color[0], label='%s AUC = %0.3f' % ('T1', roc_auc))

    fpr, tpr, threshold = metrics.roc_curve(y_test, pred2,drop_intermediate=True)
    roc_auc = metrics.auc(fpr, tpr)
    plt.plot(fpr, tpr, color[1], label='%s AUC = %0.3f' % ('T1C', roc_auc))

    fpr, tpr, threshold = metrics.roc_curve(y_test, pred3,drop_intermediate=True)
    roc_auc = metrics.auc(fpr, tpr)
    plt.plot(fpr, tpr, color[2], label='%s AUC = %0.3f' % ('T1+T1C', roc_auc))

    plt.legend(loc='lower right', fontsize=20)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.tick_params(labelsize=13)
    plt.ylabel('True Positive Rate',fontsize=Fontsize)
    plt.xlabel('False Positive Rate',fontsize=Fontsize)
    plt.show()


def Threshold(y_test, pred, pred_pro):
    pre_true=[]
    pre_false = []
    for i in range(len(y_test)):
        # if y_test[i]==1 and pred[i]==1:
        if pred[i] == 1:
            pre_true.append(pred_pro[i])
        # if y_test[i]==0 and pred[i]==0:
        if pred[i] == 0:
            pre_false.append(pred_pro[i])
    return (max(pre_false)+min(pre_true))/2

def test(excel_file = r'D:\Radiomics and Deep Learning for Identification and Automated Delineation\train_T1.xlsx',
         excel_file2 = r'D:\Radiomics and Deep Learning for Identification and Automated Delineation\test_T1.xlsx',
         excel_file3 = r'D:\Radiomics and Deep Learning for Identification and Automated Delineation\train_T1C.xlsx',
         excel_file4 = r'D:\Radiomics and Deep Learning for Identification and Automated Delineation\test_T1C.xlsx',
         feature_T1 = [105, 6, 207, 9, 841, 773, 12, 436, 479, 46],
         feature_T1C = [105, 68, 760, 755, 390, 773, 9, 206, 44, 828], model_name='SVM', plot_ROC=True):

    df = pd.read_excel(excel_file)
    df_columns = df.columns

    df2 = pd.read_excel(excel_file2)
    df_columns2 = df2.columns

    df3 = pd.read_excel(excel_file3)
    df_columns3 = df3.columns

    df4 = pd.read_excel(excel_file4)
    df_columns4 = df4.columns

    #T1
    features_origin = pd.DataFrame(df.values)
    features_origin.columns = df_columns

    features_origin2 = pd.DataFrame(df2.values)
    features_origin2.columns = df_columns2
    #T1C
    features_origin3 = pd.DataFrame(df3.values)
    features_origin3.columns = df_columns3

    features_origin4 = pd.DataFrame(df4.values)
    features_origin4.columns = df_columns4

    feature_ID = np.array(feature_T1) + 2
    feature_ID2 = np.array(feature_T1C) + 2
    #T1
    fea_number_T1 = len(feature_ID)
    tra_features1 = np.zeros(shape=(features_origin.shape[0], fea_number_T1))
    for i in range(fea_number_T1):
        tra_features1[:, i] = features_origin[features_origin.columns[feature_ID[i]]]

    tra_labels1 = features_origin[features_origin.columns[1]]

    test_features1 = np.zeros(shape=(features_origin2.shape[0], fea_number_T1))
    for i in range(fea_number_T1):
        test_features1[:, i] = features_origin2[features_origin2.columns[feature_ID[i]]]
    test_labels1 = features_origin2[features_origin2.columns[1]]

    #T1C
    fea_number_T1C = len(feature_ID2)
    tra_features2 = np.zeros(shape=(features_origin3.shape[0], fea_number_T1C))
    for i in range(fea_number_T1C):
        tra_features2[:, i] = features_origin3[features_origin3.columns[feature_ID2[i]]]

    tra_labels2 = features_origin3[features_origin3.columns[1]]

    test_features2 = np.zeros(shape=(features_origin4.shape[0], fea_number_T1C))
    for i in range(fea_number_T1C):
        test_features2[:, i] = features_origin4[features_origin4.columns[feature_ID2[i]]]

    test_labels2 = features_origin4[features_origin4.columns[1]]

    acc1 = 0
    f11 = 0
    auc1 = 0
    sp1 = 0
    rec1 = 0
    acc2 = 0
    f12 = 0
    auc2 = 0
    sp2 = 0
    rec2 = 0
    acc3 = 0
    f13 = 0
    auc3 = 0
    sp3 = 0
    rec3 = 0


    X_train1, X_test1, y_train1, y_test1 = tra_features1, test_features1, tra_labels1, test_labels1#T1
    X_train2, X_test2, y_train2, y_test2 = tra_features2, test_features2, tra_labels2, test_labels2#T1C
    stand = StandardScaler()

    X_train1 = stand.fit_transform(X_train1)
    X_test1 = stand.transform(X_test1)

    X_train2 = stand.fit_transform(X_train2)
    X_test2 = stand.transform(X_test2)

    X_train1 = np.array(X_train1)
    X_test1 = np.array(X_test1)
    y_train1= np.array(y_train1)
    y_test1 = np.array(y_test1)

    X_train2 = np.array(X_train2)
    X_test2 = np.array(X_test2)
    y_train2 = np.array(y_train2)
    y_test2 = np.array(y_test2)
    if model_name == 'RF':
        lr = RandomForestClassifier(max_depth=10, n_estimators=100, max_features='auto', criterion='gini',
                                    min_samples_leaf=2, min_samples_split=3, bootstrap=True, class_weight='balanced').fit(X_train1,
                                                                                                 y_train1)  # T1
        lr2 = RandomForestClassifier(max_depth=10, n_estimators=100, max_features='auto', criterion='gini',
                                    min_samples_leaf=1, min_samples_split=2, bootstrap=True, class_weight='balanced').fit(X_train2,
                                                                                                 y_train2)  # T1C
    if model_name == 'XGB':
        lr = XGBClassifier(eta=0.05,
                           gamma=0.2,
                           max_depth=9,
                           min_child_weight=0.0,
                           subsample=0.7,
                           colsample_bytree=0.8,
                           alpha=0.0,
                           scale_pos_weight=7).fit(X_train1, y_train1)#T1
        lr2 = XGBClassifier(eta=0.05,
                           gamma=0.2,
                           max_depth=9,
                           min_child_weight=0.0,
                           subsample=0.7,
                           colsample_bytree=0.8,
                           alpha=0.0,
                           scale_pos_weight=7).fit(X_train2, y_train2)#T1C
    if model_name == 'SVM':
        lr = SVC(C=1, cache_size=2000, class_weight='balanced', coef0=0.0,
                 decision_function_shape='ovr', degree=3, gamma='auto', kernel='linear',
                 max_iter=20000, probability=True, random_state=None, shrinking=True,
                 tol=0.001, verbose=False).fit(X_train1, y_train1)# T1
        print('*'*100)
        lr2 = SVC(C=1, cache_size=2000, class_weight='balanced', coef0=0.0,
                 decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
                 max_iter=20000, probability=True, random_state=None, shrinking=True,
                 tol=0.001, verbose=False).fit(X_train2, y_train2)# T1C

    #T1
    val_output1 = lr.predict(X_test1)
    val_score1 = compute_accuracy(val_output1, y_test1)
    val_p1 = compute_precision(val_output1, y_test1)
    val_r1 = compute_recall(val_output1, y_test1)
    val_f11 = compute_Fbeta_score(val_p1, val_r1, 1)
    val_sp1 = compute_specificity(val_output1, y_test1)
    val_auc1 = roc_auc_score(onehot_enc(y_test1), lr.predict_proba(X_test1))
    print('VAL accuracy: {:.3f}'.format(val_score1))
    print('VAL precision: {:.3f}'.format(val_p1))
    print('VAL recall: {:.3f}'.format(val_r1))
    print('VAL f1: {:.3f}'.format(val_f11))
    print('VAL specificity: {:.3f}'.format(val_sp1))
    print('VAL AUC: {:.3f}'.format(val_auc1))
    lr_pred = lr.predict_proba(X_test1)[:, 1]
    T1_threshold_value = Threshold(y_test1, val_output1, lr_pred)

    acc1 += val_score1
    f11 += val_f11
    auc1 += val_auc1
    sp1 += val_sp1
    rec1 += val_r1
    #T1C
    val_output2 = lr2.predict(X_test2)
    val_score2 = compute_accuracy(val_output2, y_test2)
    val_p2 = compute_precision(val_output2, y_test2)
    val_r2 = compute_recall(val_output2, y_test2)
    val_f12 = compute_Fbeta_score(val_p2, val_r2, 1)
    val_sp2 = compute_specificity(val_output2, y_test2)
    val_auc2 = roc_auc_score(onehot_enc(y_test2), lr2.predict_proba(X_test2))
    lr_pred2 = lr2.predict_proba(X_test2)[:, 1]
    T1C_threshold_value = Threshold(y_test2, val_output2, lr_pred2)
    print('VAL accuracy: {:.3f}'.format(val_score2))
    print('VAL precision: {:.3f}'.format(val_p2))
    print('VAL recall: {:.3f}'.format(val_r2))
    print('VAL f1: {:.3f}'.format(val_f12))
    print('VAL specificity: {:.3f}'.format(val_sp2))
    print('VAL AUC: {:.3f}'.format(val_auc2))

    acc2 += val_score2
    f12 += val_f12
    auc2 += val_auc2
    sp2 += val_sp2
    rec2 += val_r2
    #T1+T1C
    val_output_3_pro = (lr.predict_proba(X_test1) + lr2.predict_proba(X_test2)) / 2
    lr_pred3 = val_output_3_pro[:, 1]

    _threshold_value = (T1_threshold_value + T1C_threshold_value) / 2
    val_output3 = []
    for pro in val_output_3_pro:
        if pro[1] >= _threshold_value + 0.000:
            val_output3.append(1)
        else:
            val_output3.append(0)

    val_score3 = compute_accuracy(val_output3, y_test2)
    val_p3 = compute_precision(val_output3, y_test2)
    val_r3 = compute_recall(val_output3, y_test2)
    val_f13 = compute_Fbeta_score(val_p3, val_r3, 1)
    val_sp3 = compute_specificity(val_output3, y_test2)
    val_auc3 = roc_auc_score(onehot_enc(y_test2), val_output_3_pro)

    print('VAL accuracy: {:.3f}'.format(val_score3))
    print('VAL precision: {:.3f}'.format(val_p3))
    print('VAL recall: {:.3f}'.format(val_r3))
    print('VAL f1: {:.3f}'.format(val_f13))
    print('VAL specificity: {:.3f}'.format(val_sp3))
    print('VAL AUC: {:.3f}'.format(val_auc3))
    acc3 += val_score3
    f13 += val_f13
    auc3 += val_auc3
    sp3 += val_sp3
    rec3 += val_r3
    #plot ROC
    if plot_ROC==True:
        ROC_AUC_3(y_test1,lr_pred,lr_pred2,lr_pred3)

    print('T1:')
    print('Accuracy:', acc1)
    print('F1score:',f11)
    print('Specificity:',sp1)
    print('Sensitivity/Recall:',rec1)
    print('AUC:',auc1)
    print('T1C:')
    print('Accuracy:',acc2)
    print('F1score:',f12)
    print('Specificity:',sp2)
    print('Sensitivity/Recall:',rec2)
    print('AUC:',auc2)
    print('T1+T1C:')
    print('Accuracy:',acc3)
    print('F1score:',f13)
    print('Specificity:',sp3)
    print('Sensitivity/Recall:',rec3)
    print('AUC:',auc3)

if __name__ == '__main__':
    test(excel_file=r'D:\Radiomics and Deep Learning for Identification and Automated Delineation\train_T1.xlsx',
         excel_file2=r'D:\Radiomics and Deep Learning for Identification and Automated Delineation\test_T1.xlsx',
         excel_file3=r'D:\Radiomics and Deep Learning for Identification and Automated Delineation\train_T1C.xlsx',
         excel_file4=r'D:\Radiomics and Deep Learning for Identification and Automated Delineation\test_T1C.xlsx',
         feature_T1=[105, 6, 207, 9, 841, 773, 12, 436, 479, 46],
         feature_T1C=[105, 68, 760, 755, 390, 773, 9, 206, 44, 828], model_name='SVM', plot_ROC=True)