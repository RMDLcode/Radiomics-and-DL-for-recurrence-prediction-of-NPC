import pandas as pd
import numpy as np
from scipy.stats import *
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import *
from sklearn.feature_selection import *
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import *
from sklearn.preprocessing import *
from sklearn.utils import shuffle

import warnings
warnings.filterwarnings("ignore")
def onehot_enc(y):
    y = np.array(y)
    y = y.reshape(len(y), 1)
    onehot_data = OneHotEncoder(sparse=False)
    onehot_data = onehot_data.fit_transform(y)
    # onehot_data = np.flip(onehot_data, 1)
    return onehot_data

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

def read_from_excel(filename):
    df = pd.read_excel(filename)
    df = shuffle(df)
    df = df.fillna(0)

    labels = df[df.columns[2]]
    features = df[df.columns[5:]]
    return features, labels

def feature_selection(excel_file = r'D:\Radiomics and Deep Learning for Identification and Automated Delineation\train_T1.xlsx',
                      partition_min=12, partition_max=18, iterations_num=500):
    df = pd.read_excel(excel_file)
    df_columns = df.columns
    features_origin = pd.DataFrame(df.values)
    features_origin.columns = df_columns

    p = 5
    #Control max iterations_num
    k = iterations_num*10

    KF = KFold(n_splits=p)

    k_average_train_accuracy = 0.
    k_average_train_precision = 0.
    k_average_train_recall = 0.
    k_average_train_f1 = 0.
    k_average_train_auc = 0.

    k_average_val_accuracy = 0.
    k_average_val_precision = 0.
    k_average_val_recall = 0.
    k_average_val_f1 = 0.
    k_average_val_auc = 0.

    max_average_train_accuracy = 0.
    max_average_train_precision = 0.
    max_average_train_recall = 0.
    max_average_train_f1 = 0.
    max_average_train_auc = 0.

    max_average_val_accuracy = 0.
    max_average_val_precision = 0.
    max_average_val_recall = 0.
    max_average_val_f1 = 0.
    max_average_val_auc = 0.

    bad_fold = 0
    k_nn = iterations_num
    k_necessary = iterations_num
    lasso_statistic_features = np.zeros(shape=(features_origin.shape[1] - 1,)).astype(int)
    for q in range(k):
        if k_necessary <= 0:
            break
        features_ = shuffle(features_origin)
        features = features_[features_.columns[2:]]
        labels = features_[features_.columns[1]].astype(int)#label in column 1

        average_train_accuracy = 0.
        average_train_precision = 0.
        average_train_recall = 0.
        average_train_f1 = 0.
        average_train_auc = 0.

        average_val_accuracy = 0.
        average_val_precision = 0.
        average_val_recall = 0.
        average_val_f1 = 0.
        average_val_auc = 0.

        is_bad_fold = False

        for train_index, test_index in KF.split(features):
            y_train, y_test = labels.iloc[train_index], labels.iloc[test_index]
            #We want a small fluctuation in the ratio of positive and negative samples for each fold
            if np.sum(y_test) <= partition_min or np.sum(y_test) >= partition_max:
                bad_fold += 1
                is_bad_fold = True
                break
        tmp_all = []

        if not is_bad_fold:
            print(f'{k_nn - k_necessary+1}/{iterations_num}')
            k_necessary -= 1
            fold_num = 1
            for train_index, test_index in KF.split(features):
                # print("Fold %s:" % fold_num)
                fold_num += 1
                X_train, X_test = features.iloc[train_index], features.iloc[test_index]
                y_train, y_test = labels.iloc[train_index], labels.iloc[test_index]
                sc = StandardScaler()

                X_train = np.array(pd.DataFrame(sc.fit_transform(X_train)))
                X_test = np.array(pd.DataFrame(sc.transform(X_test)))

                # Lasso feature selection
                max_feature = 10
                alpha = 1e-2
                lasso = Lasso(alpha=alpha, max_iter=20000)
                lasso.fit(X_train, y_train)
                #Store selected features
                tmp_fea = np.abs(np.array(lasso.coef_))
                tmp_fea = tmp_fea.argsort()[-max_feature:][::-1]

                useful_features_number = max_feature
                for tmp in range(len(tmp_fea)):
                    if tmp < useful_features_number:
                        lasso_statistic_features[tmp_fea[tmp]] += 1

                if tmp_all == []:
                    tmp_all = tmp_fea
                else:
                    tmp_all = np.concatenate((tmp_all, tmp_fea))

                LA = SelectFromModel(lasso, prefit=True, max_features=max_feature)
                X_train = LA.transform(X_train)
                X_test = LA.transform(X_test)
                #Fit
                lr = SVC(C=1, cache_size=2000, class_weight='balanced', coef0=0.0,
                         decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
                         max_iter=30000, probability=True, random_state=None, shrinking=True,
                         tol=0.001, verbose=False).fit(X_train, y_train)
                # lr = RandomForestClassifier(max_depth=10, n_estimators=100, max_features='auto', criterion='gini',
                #                              min_samples_leaf=1, min_samples_split=2, bootstrap=True).fit(X_train, y_train)
                # lr = XGBClassifier(gamma=0.2, max_depth=9, subsample=0.7, colsample_bytree=0.8, eta=0.05,
                #                    min_child_weight=0.0, scale_pos_weight=7, alpha=0.0).fit(X_train, y_train)

                #Predict and evaluate
                #TRA
                train_output = lr.predict(X_train)
                # print('TRAIN output:', train_output)

                train_score = compute_accuracy(train_output, y_train.values)
                train_p = compute_precision(train_output, y_train.values)
                train_r = compute_recall(train_output, y_train.values)
                train_f1 = compute_Fbeta_score(train_p, train_r, 1)
                train_auc = roc_auc_score(onehot_enc(y_train), lr.predict_proba(X_train))
                # print('TRAIN accuracy: {:.3f}'.format(train_score))
                # print('TRAIN precision: {:.3f}'.format(train_p))
                # print('TRAIN recall: {:.3f}'.format(train_r))
                # print('TRAIN f1: {:.3f}'.format(train_f1))
                # print('TRAIN AUC: {:.3f}'.format(train_auc))
                average_train_accuracy += train_score / p
                average_train_precision += train_p / p
                average_train_recall += train_r / p
                average_train_f1 += train_f1 / p
                average_train_auc += train_auc / p

                #VAL
                val_output = lr.predict(X_test)
                val_score = compute_accuracy(val_output, y_test.values)
                val_p = compute_precision(val_output, y_test.values)
                val_r = compute_recall(val_output, y_test.values)
                val_f1 = compute_Fbeta_score(val_p, val_r, 1)
                val_auc = roc_auc_score(onehot_enc(y_test), lr.predict_proba(X_test))
                # print('VAL accuracy: {:.3f}'.format(val_score))
                # print('VAL precision: {:.3f}'.format(val_p))
                # print('VAL recall: {:.3f}'.format(val_r))
                # print('VAL f1: {:.3f}'.format(val_f1))
                # print('VAL AUC: {:.3f}'.format(val_auc))
                average_val_accuracy += val_score / p
                average_val_precision += val_p / p
                average_val_recall += val_r / p
                average_val_f1 += val_f1 / p
                average_val_auc += val_auc / p

            # print('\naverage_train_accuracy: {:.3f}'.format(average_train_accuracy))
            # print('average_train_precision: {:.3f}'.format(average_train_precision))
            # print('average_train_recall: {:.3f}'.format(average_train_recall))
            # print('average_train_f1: {:.3f}'.format(average_train_f1))
            # print('average_train_auc: {:.3f}'.format(average_train_auc))

            k_average_train_accuracy += average_train_accuracy
            k_average_train_precision += average_train_precision
            k_average_train_recall += average_train_recall
            k_average_train_f1 += average_train_f1
            k_average_train_auc += average_train_auc

            # print('average_val_accuracy: {:.3f}'.format(average_val_accuracy))
            # print('average_val_precision: {:.3f}'.format(average_val_precision))
            # print('average_val_recall: {:.3f}'.format(average_val_recall))
            # print('average_val_f1: {:.3f}'.format(average_val_f1))
            # print('average_val_auc: {:.3f}'.format(average_val_auc))

            k_average_val_accuracy += average_val_accuracy
            k_average_val_precision += average_val_precision
            k_average_val_recall += average_val_recall
            k_average_val_f1 += average_val_f1
            k_average_val_auc += average_val_auc
        #Save the results of the best time for understanding the upper limit
        if (2 * average_val_auc) + average_val_f1 > (2 * max_average_val_auc) + max_average_val_f1:
            max_average_train_accuracy = average_train_accuracy
            max_average_train_precision = average_train_precision
            max_average_train_recall = average_train_recall
            max_average_train_f1 = average_train_f1
            max_average_train_auc = average_train_auc

            max_average_val_accuracy = average_val_accuracy
            max_average_val_precision = average_val_precision
            max_average_val_recall = average_val_recall
            max_average_val_f1 = average_val_f1
            max_average_val_auc = average_val_auc
            # best_features = tmp_all


    frequency = k_nn - k_necessary
    k_average_train_accuracy /= frequency
    k_average_train_precision /= frequency
    k_average_train_recall /= frequency
    k_average_train_f1 /= frequency
    k_average_train_auc /= frequency

    k_average_val_accuracy /= frequency
    k_average_val_precision /= frequency
    k_average_val_recall /= frequency
    k_average_val_f1 /= frequency
    k_average_val_auc /= frequency

    print('\n' + '=' * 50 + '\n')
    k = frequency
    print('{} times {}-fold cross validation average TRAIN Accuracy: {:.3f}'.format(k, p, k_average_train_accuracy))
    print('{} times {}-fold cross validation average TRAIN Precision: {:.3f}'.format(k, p, k_average_train_precision))
    print('{} times {}-fold cross validation average TRAIN Recall: {:.3f}'.format(k, p, k_average_train_recall))
    print('{} times {}-fold cross validation average TRAIN F1: {:.3f}'.format(k, p, k_average_train_f1))
    print('{} times {}-fold cross validation average TRAIN AUC: {:.3f}'.format(k, p, k_average_train_auc))
    print('{} times {}-fold cross validation average VAL Accuracy: {:.3f}'.format(k, p, k_average_val_accuracy))
    print('{} times {}-fold cross validation average VAL Precision: {:.3f}'.format(k, p, k_average_val_precision))
    print('{} times {}-fold cross validation average VAL Recall: {:.3f}'.format(k, p, k_average_val_recall))
    print('{} times {}-fold cross validation average VAL F1: {:.3f}'.format(k, p, k_average_val_f1))
    print('{} times {}-fold cross validation average VAL AUC: {:.3f}'.format(k, p, k_average_val_auc))

    print('best {}-fold cross validation average TRAIN Accuracy: {:.3f}'.format(p, max_average_train_accuracy))
    print('best {}-fold cross validation average TRAIN Precision: {:.3f}'.format(p, max_average_train_precision))
    print('best {}-fold cross validation average TRAIN Recall: {:.3f}'.format(p, max_average_train_recall))
    print('best {}-fold cross validation average TRAIN F1: {:.3f}'.format(p, max_average_train_f1))
    print('best {}-fold cross validation average TRAIN AUC: {:.3f}'.format(p, max_average_train_auc))
    print('best {}-fold cross validation average VAL Accuracy: {:.3f}'.format(p, max_average_val_accuracy))
    print('best {}-fold cross validation average VAL Precision: {:.3f}'.format(p, max_average_val_precision))
    print('best {}-fold cross validation average VAL Recall: {:.3f}'.format(p, max_average_val_recall))
    print('best {}-fold cross validation average VAL F1: {:.3f}'.format(p, max_average_val_f1))
    print('best {}-fold cross validation average VAL AUC: {:.3f}'.format(p, max_average_val_auc))
    #top n(10) features of k(500) iterations
    Feature_ID = pd.Series(lasso_statistic_features).sort_values(ascending=False).index[:useful_features_number]
    print("The top %s features with the most frequent appearances of %s times 5-fold are as follows:" % (
    max_feature, frequency))
    for i in Feature_ID:
        print("Feature ID:", i, "Feature occurrence:", lasso_statistic_features[i], "Feature name:", df_columns[i + 2])
    print('\n')
    return Feature_ID
if __name__ == '__main__':
    Feature_ID=feature_selection(excel_file=r'D:\Radiomics and Deep Learning for Identification and Automated Delineation\train_T1C.xlsx',
                                 partition_min=12, partition_max=18, iterations_num=100)
    print(Feature_ID)
