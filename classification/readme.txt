The code is implemented in python 3.7. Other versions should work as well.
Before feature selection, it is necessary to extract radiomics features and divide data sets.
This part is the classification part of recurrence prediction.To demonstrate the process, we only provide the code for the important feature selection, training and testing procedures. 
The main function includes the feature selection function of T1 and T1C images, as well as the 5-fold cross-validation and test function after the feature selection.
We only need to ensure that the second column of the feature file in excel format is the label, and the third column and the subsequent columns are the features to achieve the entire process easily.