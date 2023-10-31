"""
================================
Recognizing hand-written digits
================================

This example shows how scikit-learn can be used to recognize images of
hand-written digits, from 0-9.

"""

# Standard scientific Python imports
import matplotlib.pyplot as plt

# Import datasets, classifiers and performance metrics
from sklearn import datasets, metrics, svm
from sklearn.model_selection import train_test_split

# Importing required functions from utils.py
from utils import preprocess_data, split_data, read_data, train_model, predict_and_eval
import pdb


# _, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
# for ax, image, label in zip(axes, digits.images, digits.target):
#     ax.set_axis_off()
#     ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
#     ax.set_title("Training: %i" % label)


X, y = read_data()

X_train, X_test, X_dev, y_train, y_test, y_dev = split_data(X, y, test_size=0.2, dev_size=0.1)

X_train = preprocess_data(X_train) 
X_test = preprocess_data(X_test) 

model = train_model(X_train, y_train, {'gamma': 0.001}, model_type='svm') 

predict_and_eval(model, X_test, y_test)