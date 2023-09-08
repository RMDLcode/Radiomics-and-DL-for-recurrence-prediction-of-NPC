import os.path
from feature_selection_lasso import feature_selection
from train import five_fold_validation
from train_test import test

folder_path = r"D:\Radiomics and Deep Learning for Identification and Automated Delineation\classification"

# We want a small fluctuation in the ratio of positive and negative samples for each fold; positive samples 12-18;
print('Feature_T1 selection:')
Feature_ID_T1 = feature_selection(excel_file=os.path.join(folder_path,r'train_T1.xlsx'), partition_min=12,
                                  partition_max=18,
                                  iterations_num=500)
print('Feature_T1C selection:')
Feature_ID_T1C = feature_selection(excel_file=os.path.join(folder_path,r'train_T1C.xlsx'), partition_min=12,
                                   partition_max=18,
                                   iterations_num=500)
# Feature_ID_T1 = [105, 6, 207, 9, 841, 773, 12, 436, 479, 46]
# Feature_ID_T1C = [105, 68, 760, 755, 390, 773, 9, 206, 44, 828]
print('5_fold_validation:')
five_fold_validation(excel_file=os.path.join(folder_path,r'train_T1.xlsx'),
                     excel_file2=os.path.join(folder_path,r'train_T1C.xlsx'),
                     feature_T1=Feature_ID_T1,
                     feature_T1C=Feature_ID_T1C,
                     partition_min=12,
                     partition_max=18,
                     plot_ROC=False
                     )

#model_name:SVM or RF or XGB
print('Test single model:')
test(excel_file=os.path.join(folder_path,r'train_T1.xlsx'),
     excel_file2=os.path.join(folder_path,r'test_T1.xlsx'),
     excel_file3=os.path.join(folder_path,r'train_T1C.xlsx'),
     excel_file4=os.path.join(folder_path,r'test_T1C.xlsx'),
     feature_T1=Feature_ID_T1,
     feature_T1C=Feature_ID_T1C,
     model_name='SVM',
     plot_ROC=False)