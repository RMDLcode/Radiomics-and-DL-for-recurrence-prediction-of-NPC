For images in nii.gz format, preprocess.py is the code that converts them to a trainable npy format.
After the conversion, we need to make a random division of the training set data, and use divide_data.py to divide the training set and the validation set.
Finally, train.py can be used to train our recurrence segmentation model.